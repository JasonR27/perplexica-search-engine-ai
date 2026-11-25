/**
 * Hybrid embedding cache decorator — T3
 * Plan: docs/plans/2026-08-29-vane-neo4j-hybrid-embedding-persistence.md
 * Wraps BaseEmbedding with dual-write: cache-first (Neo4j `memory` DB, 1024d bge-m3) + native fallback
 * AGENTS §9 strict: only 1024d bge-m3 persisted; non-1024 silently skipped
 * Non-invasive: Neo4j down → native path only, never throws
 */

import BaseEmbedding from './embedding';
import { Chunk } from '@/lib/types';
import { getCachedEmbedding, upsertEmbedding } from '@/lib/neo4j';

export class CachedEmbedding extends BaseEmbedding<any> {
  constructor(
    private readonly delegate: BaseEmbedding<any>,
    private readonly modelKey: string
  ) {
    // delegate holds real config; this wrapper has no separate config
    super((delegate as any).config);
  }

  async embedText(texts: string[]): Promise<number[][]> {
    if (!texts.length) return [];

    // Disable flag — fallback to delegate directly
    if (process.env.NEO4J_EMBEDDING_CACHE_ENABLED === 'false') {
      return this.delegate.embedText(texts);
    }

    const results: (number[] | null)[] = new Array(texts.length).fill(null);
    const missIndices: number[] = [];
    const missTexts: string[] = [];

    // Cache-first — parallel get (graceful degrade on Neo4j down)
    await Promise.all(
      texts.map(async (t, i) => {
        try {
          const cached = await getCachedEmbedding(t);
          if (cached && cached.length === 1024) {
            results[i] = cached;
          } else {
            missIndices.push(i);
            missTexts.push(t);
          }
        } catch {
          missIndices.push(i);
          missTexts.push(t);
        }
      })
    );

    // Misses → native embed
    if (missTexts.length > 0) {
      let missEmbeddings: number[][] = [];
      try {
        missEmbeddings = await this.delegate.embedText(missTexts);
      } catch (e) {
        // Delegate failure — propagate (search will handle fallback)
        throw e;
      }

      // Validate + dual-write (fire-and-forget upserts, don't block return)
      missEmbeddings.forEach((emb, j) => {
        const idx = missIndices[j];
        results[idx] = emb;
        if (emb && emb.length === 1024) {
          // Non-blocking dual-write — log but never fail search
          upsertEmbedding(missTexts[j], emb, this.modelKey).catch(() => {});
        } else if (emb && emb.length !== 0) {
          console.warn(`[embeddingCache] skip Neo4j upsert: dims ${emb.length} != 1024 (model=${this.modelKey})`);
        }
      });
    }

    return results as number[][];
  }

  async embedChunks(chunks: Chunk[]): Promise<number[][]> {
    if (!chunks.length) return [];
    const texts = chunks.map((c) => c.content);
    return this.embedText(texts);
  }
}

/**
 * Factory: wrap any BaseEmbedding with cache decorator
 * Usage: const embedding = withCache(await registry.loadEmbeddingModel(...), modelKey)
 */
export function withEmbeddingCache(
  embedding: BaseEmbedding<any>,
  modelKey: string
): BaseEmbedding<any> {
  // Already wrapped → return as-is
  if (embedding instanceof CachedEmbedding) return embedding;
  return new CachedEmbedding(embedding, modelKey);
}

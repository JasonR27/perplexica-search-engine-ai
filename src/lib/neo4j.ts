/**
 * Neo4j hybrid persistence — Vane → Neo4j `memory` DB dual-write
 * Plan: docs/plans/2026-08-29-vane-neo4j-hybrid-embedding-persistence.md T2
 * AGENTS §9: 1024d bge-m3 text strict, 2048d Qwen3-VL vision separate
 * Non-invasive: graceful degrade — Neo4j down never blocks search
 *
 * Isolated modes (per user 2026-08-29): NEO4J_CACHE_MODE flag — exclusive, no hybrid
 * - direct: bolt via neo4j-driver (perplexica → Neo4j) — isolated, no MCP hop
 * - mcp:    HTTP via vane-search-mcp gateway → MCP (perplexica → vane-search-mcp :8857 → memory DB) — isolated, no driver in perplexica
 * - auto:   definitive fallback — mcp first, driver only if gateway down (single operation, no redundancy)
 * - disabled: cache off (localStorage only)
 */

let driver: any | null = null;
let driverInitAttempted = false;
let driverAvailable = false;

type Neo4jEnv = {
  uri: string;
  user: string;
  password: string;
  database: string;
};

export type CacheMode = 'direct' | 'mcp' | 'auto' | 'disabled';

function getEnv(): Neo4jEnv {
  return {
    uri: process.env.NEO4J_URI || 'bolt://localhost:7688',
    user: process.env.NEO4J_USER || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'localdev123',
    database: process.env.NEO4J_DATABASE || 'memory',
  };
}

export function getCacheMode(): CacheMode {
  const raw = (process.env.NEO4J_CACHE_MODE || 'direct').toLowerCase();
  if (['direct', 'mcp', 'auto', 'disabled'].includes(raw)) return raw as CacheMode;
  console.warn(`[neo4j] unknown NEO4J_CACHE_MODE=${raw}, falling back to direct`);
  return 'direct';
}

function getMcpGatewayUrl(): string {
  return process.env.NEO4J_CACHE_MCP_URL || 'http://localhost:8857';
}

async function getDriver(): Promise<any | null> {
  if (driverInitAttempted) return driver;
  driverInitAttempted = true;

  const env = getEnv();
  // Feature flag — default on, but non-blocking
  if (process.env.NEO4J_EMBEDDING_CACHE_ENABLED === 'false') {
    console.log('[neo4j] Neo4j embedding cache disabled via NEO4J_EMBEDDING_CACHE_ENABLED=false');
    return null;
  }

  try {
    // Lazy import — missing dep should not crash Vane
    const neo4j = await import('neo4j-driver');
    driver = neo4j.driver(env.uri, neo4j.auth.basic(env.user, env.password), {
      maxConnectionPoolSize: 10,
      connectionAcquisitionTimeout: 5000,
    });
    // Verify connectivity without blocking search on failure
    await driver.verifyConnectivity();
    driverAvailable = true;
    console.log(`[neo4j] Connected to ${env.uri} db=${env.database}`);
    // Ensure vector index exists (best-effort, non-blocking)
    ensureVectorIndex().catch((e) => console.warn('[neo4j] ensureVectorIndex failed (non-fatal):', (e as Error).message));
    return driver;
  } catch (e: any) {
    console.warn('[neo4j] Neo4j driver init failed — embeddings will use native localStorage only:', e?.message || e);
    driver = null;
    driverAvailable = false;
    return null;
  }
}

async function ensureVectorIndex(): Promise<void> {
  const d = driver;
  if (!d) return;
  const env = getEnv();
  const session = d.session({ database: env.database });
  try {
    // Text space 1024d bge-m3 — separate from vision 2048d (correct syntax for Neo4j 2026.06 enterprise)
    await session.run(`
      CREATE VECTOR INDEX embeddingCache_idx IF NOT EXISTS
      FOR (n:EmbeddingCache) ON (n.embedding)
      OPTIONS { indexConfig: {\`vector.dimensions\`: 1024, \`vector.similarity_function\`: 'cosine'} }
    `);
  } catch (e) {
    // Index may already exist with different name/dim — non-fatal
    console.warn('[neo4j] CREATE VECTOR INDEX embeddingCache_idx skipped:', (e as Error).message);
  } finally {
    await session.close();
  }
}

function hashText(text: string): string {
  // Simple deterministic hash — avoid crypto dep, sufficient for cache key
  let h = 0;
  for (let i = 0; i < text.length; i++) {
    h = (h * 31 + text.charCodeAt(i)) | 0;
  }
  // Combine with length to reduce collision
  return `emb_${(h >>> 0).toString(36)}_${text.length}_${text.slice(0, 32).replace(/[^a-zA-Z0-9]/g, '')}`;
}

export function isNeo4jAvailable(): boolean {
  return driverAvailable && driver !== null;
}

export async function closeNeo4j(): Promise<void> {
  if (driver) {
    try {
      await driver.close();
    } catch {}
    driver = null;
    driverAvailable = false;
    driverInitAttempted = false;
  }
}

// ── Direct (bolt) provider ──────────────────────────────────────────

async function getCachedEmbeddingDirect(text: string): Promise<number[] | null> {
  const d = await getDriver();
  if (!d) return null;
  const env = getEnv();
  const id = hashText(text);
  const session = d.session({ database: env.database });
  try {
    const res = await session.run(
      `MATCH (n:EmbeddingCache {id: $id}) RETURN n.embedding as embedding, n.dimensions as dims LIMIT 1`,
      { id }
    );
    if (res.records.length === 0) return null;
    const emb = res.records[0].get('embedding');
    const dims = res.records[0].get('dims');
    if (dims !== 1024 && dims?.low !== 1024) {
      const dimVal = typeof dims === 'object' ? dims.low : dims;
      console.warn(`[neo4j:direct] cached dims ${dimVal} != 1024 — skipping`);
      return null;
    }
    if (!Array.isArray(emb) || emb.length !== 1024) return null;
    return emb as number[];
  } catch (e: any) {
    console.warn('[neo4j:direct] getCachedEmbedding failed:', e?.message || e);
    return null;
  } finally {
    await session.close();
  }
}

async function upsertEmbeddingDirect(text: string, embedding: number[], model: string): Promise<void> {
  const d = await getDriver();
  if (!d) return;
  const env = getEnv();
  const id = hashText(text);
  const session = d.session({ database: env.database });
  try {
    await session.run(
      `MERGE (n:EmbeddingCache {id: $id}) SET n.text=$text, n.embedding=$embedding, n.model=$model, n.dimensions=1024, n.updatedAt=datetime(), n.createdAt=coalesce(n.createdAt, datetime())`,
      { id, text: text.slice(0, 8000), embedding, model }
    );
  } catch (e: any) {
    console.warn('[neo4j:direct] upsert failed:', e?.message || e);
  } finally {
    await session.close();
  }
}

// ── MCP proxy provider (perplexica → vane-search-mcp :8857 → NAMS/Neo4j) ──

async function getCachedEmbeddingMcp(text: string): Promise<number[] | null> {
  const url = `${getMcpGatewayUrl()}/cache/embedding/get`;
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 2000);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, id: hashText(text) }),
      signal: ctrl.signal,
    });
  } catch (e: any) {
    clearTimeout(t);
    // gateway down — throw for definitive fallback, not silent miss
    throw new Error(`[neo4j:mcp] gateway unreachable: ${e?.message || e}`);
  }
  clearTimeout(t);
  if (!res.ok) throw new Error(`[neo4j:mcp] GET ${res.status}`);
  const data: any = await res.json();
  const emb = data?.embedding;
  const dims = data?.dimensions;
  if (!emb) return null; // cache miss — not an error, don't fallback
  if (dims !== 1024) {
    console.warn(`[neo4j:mcp] dims ${dims} != 1024 — skipping`);
    return null;
  }
  if (!Array.isArray(emb) || emb.length !== 1024) return null;
  return emb as number[];
}

async function upsertEmbeddingMcp(text: string, embedding: number[], model: string): Promise<void> {
  const url = `${getMcpGatewayUrl()}/cache/embedding/set`;
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 2000);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, id: hashText(text), embedding, model, dimensions: 1024 }),
      signal: ctrl.signal,
    });
  } catch (e: any) {
    clearTimeout(t);
    throw new Error(`[neo4j:mcp] gateway unreachable: ${e?.message || e}`);
  }
  clearTimeout(t);
  if (!res.ok) throw new Error(`[neo4j:mcp] SET ${res.status}`);
}

// ── Facade — mode switch via NEO4J_CACHE_MODE ───────────────────────

export async function getCachedEmbedding(text: string): Promise<number[] | null> {
  if (process.env.NEO4J_EMBEDDING_CACHE_ENABLED === 'false') return null;
  const mode = getCacheMode();
  if (mode === 'disabled') return null;
  if (mode === 'direct') return getCachedEmbeddingDirect(text);
  if (mode === 'mcp') return getCachedEmbeddingMcp(text);
  if (mode === 'auto') {
    // definitive fallback — single operation: mcp first, driver only if gateway down
    try {
      return await getCachedEmbeddingMcp(text);
    } catch (e: any) {
      console.warn(`[neo4j:auto] mcp get failed, fallback to direct: ${e.message}`);
      return getCachedEmbeddingDirect(text);
    }
  }
  return getCachedEmbeddingDirect(text);
}

export async function upsertEmbedding(text: string, embedding: number[], model: string = 'bge-m3:latest'): Promise<void> {
  if (!Array.isArray(embedding) || embedding.length !== 1024) {
    if (embedding.length !== 0) console.warn(`[neo4j] skip upsert: dims ${embedding.length} !=1024 model=${model}`);
    return;
  }
  if (process.env.NEO4J_EMBEDDING_CACHE_ENABLED === 'false') return;
  const mode = getCacheMode();
  if (mode === 'disabled') return;
  if (mode === 'direct') return upsertEmbeddingDirect(text, embedding, model);
  if (mode === 'mcp') return upsertEmbeddingMcp(text, embedding, model);
  if (mode === 'auto') {
    // definitive fallback — single write: mcp first, driver only if gateway down (no dual-write)
    try {
      await upsertEmbeddingMcp(text, embedding, model);
      return;
    } catch (e: any) {
      console.warn(`[neo4j:auto] mcp set failed, fallback to direct: ${e.message}`);
      return upsertEmbeddingDirect(text, embedding, model);
    }
  }
}

// Eager init on import (non-blocking) — warm connection early
getDriver().catch(() => {});

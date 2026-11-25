import ModelRegistry from '@/lib/models/registry';
import BaseLLM from '@/lib/models/base/llm';
import { ModelWithProvider } from '@/lib/models/types';
import SessionManager from '@/lib/session';
import { ChatTurnMessage } from '@/lib/types';
import {
  SearchInferenceRole,
  SearchRoleLLMs,
  SearchSources,
} from '@/lib/agents/search/types';
import APISearchAgent from '@/lib/agents/search/api';

interface ChatRequestBody {
  optimizationMode: 'speed' | 'balanced' | 'quality';
  sources: SearchSources[];
  chatModel: ModelWithProvider;
  inferenceModels?: Partial<Record<SearchInferenceRole, ModelWithProvider>>;
  embeddingModel: ModelWithProvider;
  query: string;
  history: Array<[string, string]>;
  stream?: boolean;
  systemInstructions?: string;
}

export const POST = async (req: Request) => {
  try {
    const body: ChatRequestBody = await req.json();

    if (!body.sources || !body.query) {
      return Response.json(
        { message: 'Missing sources or query' },
        { status: 400 },
      );
    }

    body.history = body.history || [];
    body.optimizationMode = body.optimizationMode || 'speed';
    body.stream = body.stream || false;

    const registry = new ModelRegistry();
    const llmCache = new Map<string, Promise<BaseLLM<any>>>();
    const loadCachedChatModel = (spec: ModelWithProvider) => {
      const cacheKey = `${spec.providerId}::${spec.key}`;
      let pending = llmCache.get(cacheKey);
      if (!pending) {
        pending = registry.loadChatModel(spec.providerId, spec.key);
        llmCache.set(cacheKey, pending);
      }
      return pending;
    };

    const roles: SearchInferenceRole[] = [
      'classifier',
      'widget',
      'researcher',
      'picker',
      'extractor',
      'writer',
    ];
    const resolvedSpecs: Record<SearchInferenceRole, ModelWithProvider> = {
      classifier: body.inferenceModels?.classifier ?? body.chatModel,
      widget: body.inferenceModels?.widget ?? body.chatModel,
      researcher: body.inferenceModels?.researcher ?? body.chatModel,
      picker: body.inferenceModels?.picker ?? body.chatModel,
      extractor: body.inferenceModels?.extractor ?? body.chatModel,
      writer: body.inferenceModels?.writer ?? body.chatModel,
    };

    const [llm, embeddings, ...loadedRoleModels] = await Promise.all([
      loadCachedChatModel(body.chatModel),
      registry.loadEmbeddingModel(
        body.embeddingModel.providerId,
        body.embeddingModel.key,
      ),
      ...roles.map((r) => loadCachedChatModel(resolvedSpecs[r])),
    ]);

    const roleLLMs: SearchRoleLLMs = {};
    roles.forEach((role, idx) => {
      roleLLMs[role] = loadedRoleModels[idx];
    });

    const history: ChatTurnMessage[] = body.history.map((msg) => {
      return msg[0] === 'human'
        ? { role: 'user', content: msg[1] }
        : { role: 'assistant', content: msg[1] };
    });

    const session = SessionManager.createSession();

    const agent = new APISearchAgent();
    const startSearch = () => {
      void agent
        .searchAsync(session, {
          chatHistory: history,
          config: {
            embedding: embeddings,
            llm: llm,
            roleLLMs,
            sources: body.sources,
            mode: body.optimizationMode,
            fileIds: [],
            systemInstructions: body.systemInstructions || '',
          },
          followUp: body.query,
          chatId: crypto.randomUUID(),
          messageId: crypto.randomUUID(),
        })
        .catch((err: any) => {
          const status = Number(err?.status ?? err?.statusCode ?? 500);
          session.emit('error', {
            message: err?.message || String(err),
            status: status >= 400 && status <= 599 ? status : 500,
          });
        });
    };

    if (!body.stream) {
      return new Promise((resolve: (value: Response) => void) => {
        let message = '';
        let sources: any[] = [];
        let settled = false;

        session.subscribe((event: string, data: Record<string, any>) => {
          if (settled) return;

          if (event === 'data') {
            try {
              if (data.type === 'response') {
                message += data.data;
              } else if (data.type === 'searchResults') {
                sources = data.data;
              }
            } catch (error: any) {
              settled = true;
              resolve(
                Response.json(
                  { message: error?.message || 'Error parsing data' },
                  { status: 500 },
                ),
              );
            }
          }

          if (event === 'end') {
            settled = true;
            resolve(
              Response.json(
                {
                  message,
                  sources,
                  inferenceModelsUsed: resolvedSpecs,
                },
                { status: 200 },
              ),
            );
          }

          if (event === 'error') {
            settled = true;
            const status =
              typeof data?.status === 'number' &&
              data.status >= 400 &&
              data.status <= 599
                ? data.status
                : 500;
            resolve(
              Response.json(
                {
                  message: data?.message || 'Search error',
                  error: data,
                },
                { status },
              ),
            );
          }
        });

        startSearch();
      });
    }

    const encoder = new TextEncoder();

    const abortController = new AbortController();
    const { signal } = abortController;

    const stream = new ReadableStream({
      start(controller) {
        let sources: any[] = [];

        controller.enqueue(
          encoder.encode(
            JSON.stringify({
              type: 'init',
              data: 'Stream connected',
            }) + '\n',
          ),
        );

        signal.addEventListener('abort', () => {
          session.removeAllListeners();

          try {
            controller.close();
          } catch (error) {}
        });

        session.subscribe((event: string, data: Record<string, any>) => {
          if (event === 'data') {
            if (signal.aborted) return;

            try {
              if (data.type === 'response') {
                controller.enqueue(
                  encoder.encode(
                    JSON.stringify({
                      type: 'response',
                      data: data.data,
                    }) + '\n',
                  ),
                );
              } else if (data.type === 'searchResults') {
                sources = data.data;
                controller.enqueue(
                  encoder.encode(
                    JSON.stringify({
                      type: 'sources',
                      data: sources,
                    }) + '\n',
                  ),
                );
              }
            } catch (error) {
              controller.error(error);
            }
          }

          if (event === 'end') {
            if (signal.aborted) return;

            controller.enqueue(
              encoder.encode(
                JSON.stringify({
                  type: 'done',
                }) + '\n',
              ),
            );
            controller.close();
          }

          if (event === 'error') {
            if (signal.aborted) return;

            controller.error(data);
          }
        });
      },
      cancel() {
        abortController.abort();
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
      },
    });
  } catch (err: any) {
    console.error(`Error in getting search results: ${err.message}`);
    return Response.json(
      { message: 'An error has occurred.' },
      { status: 500 },
    );
  }
};

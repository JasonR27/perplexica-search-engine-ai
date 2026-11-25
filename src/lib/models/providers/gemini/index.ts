import { existsSync, readFileSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import { UIConfigField } from '@/lib/config/types';
import { getConfiguredModelProviderById } from '@/lib/config/serverRegistry';
import { Model, ModelList, ProviderMetadata } from '../../types';
import GeminiEmbedding from './geminiEmbedding';
import BaseEmbedding from '../../base/embedding';
import BaseModelProvider from '../../base/provider';
import BaseLLM from '../../base/llm';
import GeminiLLM from './geminiLLM';

interface GeminiConfig {
  apiKey: string;
}

function resolveGeminiSecret(value: unknown): string {
  const raw = String(value ?? '');
  const match = raw.match(/^\$\{([A-Z0-9_]+)\}$/);
  if (!match) return raw;
  const envName = match[1];
  if (process.env[envName]) return process.env[envName]!;
  try {
    const envPath = join(homedir(), '.config', 'opencode', '.env');
    if (existsSync(envPath)) {
      for (const line of readFileSync(envPath, 'utf8').split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const eq = trimmed.indexOf('=');
        if (eq > 0 && trimmed.slice(0, eq).trim() === envName) {
          return trimmed.slice(eq + 1).trim().replace(/^['"]|['"]$/g, '');
        }
      }
    }
  } catch {}
  return '';
}

const providerConfigFields: UIConfigField[] = [
  {
    type: 'password',
    name: 'API Key',
    key: 'apiKey',
    description: 'Your Gemini API key',
    required: true,
    placeholder: 'Gemini API Key',
    env: 'GEMINI_API_KEY',
    scope: 'server',
  },
];

class GeminiProvider extends BaseModelProvider<GeminiConfig> {
  constructor(id: string, name: string, config: GeminiConfig) {
    super(id, name, {
      ...config,
      apiKey: resolveGeminiSecret(config?.apiKey),
    });
  }

  async getDefaultModels(): Promise<ModelList> {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models?key=${this.config.apiKey}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      },
    );

    const data = await res.json();

    let defaultEmbeddingModels: Model[] = [];
    let defaultChatModels: Model[] = [];

    data.models.forEach((m: any) => {
      if (
        m.supportedGenerationMethods.some(
          (genMethod: string) =>
            genMethod === 'embedText' || genMethod === 'embedContent',
        )
      ) {
        defaultEmbeddingModels.push({
          key: m.name,
          name: m.displayName,
        });
      } else if (m.supportedGenerationMethods.includes('generateContent')) {
        defaultChatModels.push({
          key: m.name,
          name: m.displayName,
        });
      }
    });

    return {
      embedding: defaultEmbeddingModels,
      chat: defaultChatModels,
    };
  }

  async getModelList(): Promise<ModelList> {
    const defaultModels = await this.getDefaultModels();
    const configProvider = getConfiguredModelProviderById(this.id)!;

    // Reconciled 2026-10-30 Free-Tier HTTP-200 + FULL-contract Gemini models
    // (excludes overloaded 503 gemini-3.8-flash, 35s-timeout gemini-3.7-flash/gemma-4,
    // 404 retired gemini-2.5-*, and 429 limit:0 paid-only models).
    const ALLOWED_GEMINI_KEYS = [
      'models/gemini-3.5-flash-lite',
      'models/gemini-3.1-flash-lite',
      'models/gemini-3.6-flash',
      'models/gemini-3-flash-preview',
      'models/gemini-flash-lite-latest',
      'models/gemini-3.1-flash-lite-preview',
      'models/gemini-3.5-flash',
    ];

    const filteredChat = ALLOWED_GEMINI_KEYS.map((allowedKey) =>
      defaultModels.chat.find((m) => m.key === allowedKey),
    ).filter((m): m is Model => Boolean(m));

    // Deduplicate while preserving verified priority order
    const seen = new Set<string>();
    const dedupedChat = [
      ...filteredChat,
      ...configProvider.chatModels.filter((m) =>
        ALLOWED_GEMINI_KEYS.includes(m.key),
      ),
    ].filter((m) => {
      if (seen.has(m.key)) return false;
      seen.add(m.key);
      return true;
    });

    return {
      embedding: [
        ...defaultModels.embedding,
        ...configProvider.embeddingModels,
      ],
      chat: dedupedChat,
    };
  }

  async loadChatModel(key: string): Promise<BaseLLM<any>> {
    const normalizedKey = key.startsWith('models/') ? key : `models/${key}`;
    const openAiModelId = key.replace(/^models\//, '');
    const modelList = await this.getModelList();

    const exists = modelList.chat.find(
      (m) => m.key === normalizedKey || m.key === openAiModelId,
    );

    if (!exists) {
      throw new Error(
        'Error Loading Gemini Chat Model. Invalid Model Selected',
      );
    }

    // CRITICAL: Google's /v1beta/openai/chat/completions requires bare model IDs
    // (e.g. 'gemini-3.5-flash-lite'); passing 'models/...' hangs until read timeout.
    return new GeminiLLM({
      apiKey: this.config.apiKey,
      model: openAiModelId,
      baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
    });
  }

  async loadEmbeddingModel(key: string): Promise<BaseEmbedding<any>> {
    const normalizedKey = key.startsWith('models/') ? key : `models/${key}`;
    const openAiModelId = key.replace(/^models\//, '');
    const modelList = await this.getModelList();
    const exists = modelList.embedding.find(
      (m) => m.key === normalizedKey || m.key === openAiModelId,
    );

    if (!exists) {
      throw new Error(
        'Error Loading Gemini Embedding Model. Invalid Model Selected.',
      );
    }

    return new GeminiEmbedding({
      apiKey: this.config.apiKey,
      model: openAiModelId,
      baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
    });
  }

  static parseAndValidate(raw: any): GeminiConfig {
    if (!raw || typeof raw !== 'object')
      throw new Error('Invalid config provided. Expected object');
    const resolvedKey = resolveGeminiSecret(raw.apiKey);
    if (!resolvedKey)
      throw new Error('Invalid config provided. API key must be provided');

    return {
      apiKey: resolvedKey,
    };
  }

  static getProviderConfigFields(): UIConfigField[] {
    return providerConfigFields;
  }

  static getProviderMetadata(): ProviderMetadata {
    return {
      key: 'gemini',
      name: 'Gemini',
    };
  }
}

export default GeminiProvider;

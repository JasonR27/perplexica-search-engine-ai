import { existsSync, readFileSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import { UIConfigField } from '@/lib/config/types';
import { getConfiguredModelProviderById } from '@/lib/config/serverRegistry';
import { Model, ModelList, ProviderMetadata } from '../../types';
import OpenAIEmbedding from './openaiEmbedding';
import BaseEmbedding from '../../base/embedding';
import BaseModelProvider from '../../base/provider';
import BaseLLM from '../../base/llm';
import OpenAILLM from './openaiLLM';

interface OpenAIConfig {
  apiKey: string;
  baseURL: string;
}

function resolveConfiguredSecret(value: unknown): string {
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

const defaultChatModels: Model[] = [
  {
    name: 'GPT-3.5 Turbo',
    key: 'gpt-3.5-turbo',
  },
  {
    name: 'GPT-4',
    key: 'gpt-4',
  },
  {
    name: 'GPT-4 turbo',
    key: 'gpt-4-turbo',
  },
  {
    name: 'GPT-4 omni',
    key: 'gpt-4o',
  },
  {
    name: 'GPT-4o (2024-05-13)',
    key: 'gpt-4o-2024-05-13',
  },
  {
    name: 'GPT-4 omni mini',
    key: 'gpt-4o-mini',
  },
  {
    name: 'GPT 4.1 nano',
    key: 'gpt-4.1-nano',
  },
  {
    name: 'GPT 4.1 mini',
    key: 'gpt-4.1-mini',
  },
  {
    name: 'GPT 4.1',
    key: 'gpt-4.1',
  },
  {
    name: 'GPT 5 nano',
    key: 'gpt-5-nano',
  },
  {
    name: 'GPT 5',
    key: 'gpt-5',
  },
  {
    name: 'GPT 5 Mini',
    key: 'gpt-5-mini',
  },
  {
    name: 'GPT 5 Pro',
    key: 'gpt-5-pro',
  },
  {
    name: 'GPT 5.1',
    key: 'gpt-5.1',
  },
  {
    name: 'GPT 5.2',
    key: 'gpt-5.2',
  },
  {
    name: 'GPT 5.2 Pro',
    key: 'gpt-5.2-pro',
  },
  {
    name: 'o1',
    key: 'o1',
  },
  {
    name: 'o3',
    key: 'o3',
  },
  {
    name: 'o3 Mini',
    key: 'o3-mini',
  },
  {
    name: 'o4 Mini',
    key: 'o4-mini',
  },
];

const defaultEmbeddingModels: Model[] = [
  {
    name: 'Text Embedding 3 Small',
    key: 'text-embedding-3-small',
  },
  {
    name: 'Text Embedding 3 Large',
    key: 'text-embedding-3-large',
  },
];

const providerConfigFields: UIConfigField[] = [
  {
    type: 'password',
    name: 'API Key',
    key: 'apiKey',
    description: 'Your OpenAI API key',
    required: true,
    placeholder: 'OpenAI API Key',
    env: 'OPENAI_API_KEY',
    scope: 'server',
  },
  {
    type: 'string',
    name: 'Base URL',
    key: 'baseURL',
    description: 'The base URL for the OpenAI API',
    required: true,
    placeholder: 'OpenAI Base URL',
    default: 'https://api.openai.com/v1',
    env: 'OPENAI_BASE_URL',
    scope: 'server',
  },
];

class OpenAIProvider extends BaseModelProvider<OpenAIConfig> {
  constructor(id: string, name: string, config: OpenAIConfig) {
    super(id, name, {
      ...config,
      apiKey: resolveConfiguredSecret(config?.apiKey),
    });
  }

  async getDefaultModels(): Promise<ModelList> {
    if (this.config.baseURL === 'https://api.openai.com/v1') {
      return {
        embedding: defaultEmbeddingModels,
        chat: defaultChatModels,
      };
    }

    return {
      embedding: [],
      chat: [],
    };
  }

  async getModelList(): Promise<ModelList> {
    const defaultModels = await this.getDefaultModels();

    // Reconciled 2026-10-29 NVIDIA NIM HTTP-200 verified models first;
    // 60s-timeout preview slugs kept only at tail for manual testing.
    const NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1';
    const NIM_CHAT_MODELS: Model[] = [
      { name: 'Nemotron 3.5 Bolt E4B', key: 'nvidia/nemotron-3.5-bolt-e4b' },
      { name: 'Nemotron 3.5 Lightning 30B-A3B', key: 'nvidia/nemotron-3.5-lightning-30b-a3b' },
      { name: 'Mistral Large 4 1T-A52B', key: 'mistralai/mistral-large-4-1t-a52b' },
      { name: 'Nemotron 3 Ultra 550B-A55B', key: 'nvidia/nemotron-3-ultra-550b-a55b' },
      { name: 'Nemotron 3.5 Super VL 120B-A12B', key: 'nvidia/nemotron-3.5-super-vl-120b-a12b' },
      { name: 'MiniMax M3.1 Flash', key: 'minimaxai/minimax-m3.1-flash' },
      { name: 'Meta Muse Glimmer 30B', key: 'meta/muse-glimmer-30b' },
      { name: 'Llama 3.2 11B Vision Instruct', key: 'meta/llama-3.2-11b-vision-instruct' },
      { name: 'GPT-OSS 20B (NIM)', key: 'openai/gpt-oss-20b' },
    ];

    if (this.config.baseURL === NIM_BASE_URL) {
      return {
        embedding: [],
        chat: NIM_CHAT_MODELS,
      };
    }

    // Reconciled 2026-10-29 OpenRouter zero-price HTTP-200 chat/tool models
    // (excludes nemotron-3.5-content-safety:free guard, paid lyria, and 403 inkling).
    const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
    const OPENROUTER_FREE_CHAT_MODELS: Model[] = [
      { name: 'Nemotron 3.5 Super VL Free', key: 'nvidia/nemotron-3.5-super-vl:free' },
      { name: 'Nemotron 3.5 Lightning Free', key: 'nvidia/nemotron-3.5-lightning:free' },
      { name: 'Nemotron 3 Ultra 550B Free', key: 'nvidia/nemotron-3-ultra-550b-a55b:free' },
      { name: 'DOTS 3 Note Preview Free', key: 'dots-studio/dots-3-note-preview:free' },
      { name: 'Xing 4.0 29B-A4B Free', key: 'xingchen-agi/xing4.0-29b-a4b:free' },
      { name: 'Nemotron 3 Nano Omni 30B Free', key: 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free' },
      { name: 'Laguna S 2.1 Free', key: 'poolside/laguna-s-2.1:free' },
      { name: 'Laguna XS 2.1 Free', key: 'poolside/laguna-xs-2.1:free' },
      { name: 'Cohere North Mini Code Free', key: 'cohere/north-mini-code:free' },
      { name: 'Liquid LFM 2.5 2.6B Free', key: 'liquid/lfm-2.5-2.6b:free' },
      { name: 'Flint v1.0 Free', key: 'recursive/flint-v1.0:free' },
      { name: 'OpenRouter Free Router', key: 'openrouter/free' },
    ];

    const configProvider = getConfiguredModelProviderById(this.id)!;

    if (this.config.baseURL === OPENROUTER_BASE_URL) {
      const seen = new Set<string>();
      const merged = [
        ...OPENROUTER_FREE_CHAT_MODELS,
        ...(configProvider?.chatModels ?? []),
      ].filter((m) => {
        if (seen.has(m.key)) return false;
        seen.add(m.key);
        return true;
      });
      return {
        embedding: [],
        chat: merged,
      };
    }

    return {
      embedding: [
        ...defaultModels.embedding,
        ...configProvider.embeddingModels,
      ],
      chat: [...defaultModels.chat, ...configProvider.chatModels],
    };
  }

  async loadChatModel(key: string): Promise<BaseLLM<any>> {
    const modelList = await this.getModelList();

    const exists = modelList.chat.find((m) => m.key === key);

    if (!exists) {
      throw new Error(
        'Error Loading OpenAI Chat Model. Invalid Model Selected',
      );
    }

    return new OpenAILLM({
      apiKey: this.config.apiKey,
      model: key,
      baseURL: this.config.baseURL,
    });
  }

  async loadEmbeddingModel(key: string): Promise<BaseEmbedding<any>> {
    const modelList = await this.getModelList();
    const exists = modelList.embedding.find((m) => m.key === key);

    if (!exists) {
      throw new Error(
        'Error Loading OpenAI Embedding Model. Invalid Model Selected.',
      );
    }

    return new OpenAIEmbedding({
      apiKey: this.config.apiKey,
      model: key,
      baseURL: this.config.baseURL,
    });
  }

  static parseAndValidate(raw: any): OpenAIConfig {
    if (!raw || typeof raw !== 'object')
      throw new Error('Invalid config provided. Expected object');
    if (!raw.apiKey || !raw.baseURL)
      throw new Error(
        'Invalid config provided. API key and base URL must be provided',
      );

    return {
      apiKey: resolveConfiguredSecret(raw.apiKey),
      baseURL: String(raw.baseURL),
    };
  }

  static getProviderConfigFields(): UIConfigField[] {
    return providerConfigFields;
  }

  static getProviderMetadata(): ProviderMetadata {
    return {
      key: 'openai',
      name: 'OpenAI',
    };
  }
}

export default OpenAIProvider;

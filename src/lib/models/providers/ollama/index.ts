import { UIConfigField } from '@/lib/config/types';
import { getConfiguredModelProviderById } from '@/lib/config/serverRegistry';
import BaseModelProvider from '../../base/provider';
import { Model, ModelList, ProviderMetadata } from '../../types';
import BaseLLM from '../../base/llm';
import BaseEmbedding from '../../base/embedding';
import OllamaLLM from './ollamaLLM';
import OllamaEmbedding from './ollamaEmbedding';

interface OllamaConfig {
  baseURL: string;
}

const providerConfigFields: UIConfigField[] = [
  {
    type: 'string',
    name: 'Base URL',
    key: 'baseURL',
    description: 'The base URL for the Ollama',
    required: true,
    placeholder: process.env.DOCKER
      ? 'http://host.docker.internal:11434'
      : 'http://localhost:11434',
    env: 'OLLAMA_BASE_URL',
    scope: 'server',
  },
];

class OllamaProvider extends BaseModelProvider<OllamaConfig> {
  constructor(id: string, name: string, config: OllamaConfig) {
    super(id, name, config);
  }

  // STRICT per AGENTS.md §9: 1024d = BAAI/bge-m3 ONLY; 2048d = Qwen3-VL-Embedding-2B ONLY.
  // OLLAMA CLOUD FIRST: :cloud models run on Ollama's free datacenter GPU (weekly limit),
  // always preferred over local M2200 4GB. Fallback to local ONLY when cloud quota exhausted.
  // Autodiscover is DISABLED from unfiltered /api/tags — we expose ONLY allowlisted models.
  private static readonly ALLOWED_EMBEDDING_KEYS = ['bge-m3:latest', 'bge-m3', 'BAAI/bge-m3'];

  // Free Ollama cloud models (:cloud suffix) — gpt-oss:120b-cloud > nemotron-3-super:cloud > nemotron-nano > gemma4:31b-cloud > local
  // Local GPU models as last-resort fallbacks only
  private static readonly CLOUD_CHAT_KEYS = [
    'gpt-oss:120b-cloud',          // Best free cloud (120B params, Ollama datacenter GPU)
    'gpt-oss:20b-cloud',           // Smaller cloud alternative
    'nemotron-3-super:cloud',      // NVIDIA 120B A12B on cloud (matches NIM nemotron-3-super)
    'nemotron-3-nano:30b-cloud',   // Fast 30B reasoning (matches NIM nemotron-3-nano)
    'gemma4:31b-cloud',            // Gemma 31B on cloud (free tier)
  ];
  private static readonly LOCAL_CHAT_KEYS = [
    'granite4.1:3b',
    'qwen3.5:4b',
    'phi4-mini:latest',
    'gemma3:4b',
  ];

  async getDefaultModels(): Promise<ModelList> {
    try {
      const res = await fetch(`${this.config.baseURL}/api/tags`, {
        method: 'GET',
        headers: {
          'Content-type': 'application/json',
        },
      });

      const data = await res.json();

      const allModels: Model[] = data.models.map((m: any) => {
        return {
          name: m.name,
          key: m.model,
        };
      });

      // STRICT allowlist filter — autodiscover disabled for non-permitted models/dims
      const embedding = allModels.filter((m) => {
        const k = m.key.toLowerCase();
        return OllamaProvider.ALLOWED_EMBEDDING_KEYS.some((a) => k === a.toLowerCase() || k.includes(a.toLowerCase()));
      });

      // CLOUD-FIRST SORT: :cloud models (free Ollama datacenter GPU) before local GPU models
      const cloudChat = allModels.filter((m) => {
        const k = m.key.toLowerCase();
        return OllamaProvider.CLOUD_CHAT_KEYS.some((ck) => k === ck.toLowerCase() || k.includes(ck.toLowerCase()));
      });
      const localChat = allModels.filter((m) => {
        const k = m.key.toLowerCase();
        return OllamaProvider.LOCAL_CHAT_KEYS.some((lk) => k === lk.toLowerCase());
      });

      // Cloud first, then local — local ONLY if cloud quota exhausted
      return {
        embedding,
        chat: [...cloudChat, ...localChat],
      };
    } catch (err) {
      if (err instanceof TypeError) {
        throw new Error(
          'Error connecting to Ollama API. Please ensure the base URL is correct and the Ollama server is running.',
        );
      }

      throw err;
    }
  }

  async getModelList(): Promise<ModelList> {
    const defaultModels = await this.getDefaultModels();
    const configProvider = getConfiguredModelProviderById(this.id);

    const installedChatKeys = new Set(defaultModels.chat.map((m) => m.key));
    const seenChat = new Set<string>();
    const dedupedChat = [
      ...defaultModels.chat,
      ...(configProvider?.chatModels ?? []).filter((m) =>
        installedChatKeys.has(m.key),
      ),
    ].filter((m) => {
      if (seenChat.has(m.key)) return false;
      seenChat.add(m.key);
      return true;
    });

    const seenEmbed = new Set<string>();
    const dedupedEmbed = [
      ...defaultModels.embedding,
      ...(configProvider?.embeddingModels ?? []),
    ].filter((m) => {
      if (seenEmbed.has(m.key)) return false;
      seenEmbed.add(m.key);
      return true;
    });

    return {
      embedding: dedupedEmbed,
      chat: dedupedChat,
    };
  }

  async loadChatModel(key: string): Promise<BaseLLM<any>> {
    const modelList = await this.getModelList();

    const exists = modelList.chat.find((m) => m.key === key);

    if (!exists) {
      throw new Error(
        'Error Loading Ollama Chat Model. Invalid Model Selected',
      );
    }

    return new OllamaLLM({
      baseURL: this.config.baseURL,
      model: key,
    });
  }

  async loadEmbeddingModel(key: string): Promise<BaseEmbedding<any>> {
    const modelList = await this.getModelList();
    const exists = modelList.embedding.find((m) => m.key === key);

    if (!exists) {
      throw new Error(
        'Error Loading Ollama Embedding Model. Invalid Model Selected.',
      );
    }

    return new OllamaEmbedding({
      model: key,
      baseURL: this.config.baseURL,
    });
  }

  static parseAndValidate(raw: any): OllamaConfig {
    if (!raw || typeof raw !== 'object')
      throw new Error('Invalid config provided. Expected object');
    if (!raw.baseURL)
      throw new Error('Invalid config provided. Base URL must be provided');

    return {
      baseURL: String(raw.baseURL),
    };
  }

  static getProviderConfigFields(): UIConfigField[] {
    return providerConfigFields;
  }

  static getProviderMetadata(): ProviderMetadata {
    return {
      key: 'ollama',
      name: 'Ollama',
    };
  }
}

export default OllamaProvider;

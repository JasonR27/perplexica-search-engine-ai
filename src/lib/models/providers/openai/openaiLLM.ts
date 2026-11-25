import OpenAI from 'openai';
import BaseLLM from '../../base/llm';
import { zodTextFormat, zodResponseFormat } from 'openai/helpers/zod';
import {
  GenerateObjectInput,
  GenerateOptions,
  GenerateTextInput,
  GenerateTextOutput,
  StreamTextOutput,
  ToolCall,
} from '../../types';
import { parse } from 'partial-json';
import z from 'zod';
import {
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolMessageParam,
} from 'openai/resources/index.mjs';
import { Message } from '@/lib/types';
import { repairJson } from '@toolsycc/json-repair';

type OpenAIConfig = {
  apiKey: string;
  model: string;
  baseURL?: string;
  options?: GenerateOptions;
};

class OpenAILLM extends BaseLLM<OpenAIConfig> {
  openAIClient: OpenAI;

  constructor(protected config: OpenAIConfig) {
    super(config);

    this.openAIClient = new OpenAI({
      apiKey: this.config.apiKey,
      baseURL: this.config.baseURL || 'https://api.openai.com/v1',
    });
  }

  convertToOpenAIMessages(messages: Message[]): ChatCompletionMessageParam[] {
    return messages.map((msg) => {
      if (msg.role === 'tool') {
        return {
          role: 'tool',
          tool_call_id: msg.id,
          content: msg.content,
        } as ChatCompletionToolMessageParam;
      } else if (msg.role === 'assistant') {
        return {
          role: 'assistant',
          content: msg.content,
          ...(msg.tool_calls &&
            msg.tool_calls.length > 0 && {
              tool_calls: msg.tool_calls?.map((tc) => ({
                id: tc.id,
                type: 'function',
                function: {
                  name: tc.name,
                  arguments: JSON.stringify(tc.arguments),
                },
              })),
            }),
        } as ChatCompletionAssistantMessageParam;
      }

      return msg;
    });
  }

  async generateText(input: GenerateTextInput): Promise<GenerateTextOutput> {
    const openaiTools: ChatCompletionTool[] = [];

    input.tools?.forEach((tool) => {
      openaiTools.push({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: z.toJSONSchema(tool.schema),
        },
      });
    });

    const response = await this.openAIClient.chat.completions.create({
      model: this.config.model,
      tools: openaiTools.length > 0 ? openaiTools : undefined,
      messages: this.convertToOpenAIMessages(input.messages),
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 1.0,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
    });

    if (response.choices && response.choices.length > 0) {
      return {
        content: response.choices[0].message.content!,
        toolCalls:
          response.choices[0].message.tool_calls
            ?.map((tc) => {
              if (tc.type === 'function') {
                return {
                  name: tc.function.name,
                  id: tc.id,
                  arguments: JSON.parse(tc.function.arguments),
                };
              }
            })
            .filter((tc) => tc !== undefined) || [],
        additionalInfo: {
          finishReason: response.choices[0].finish_reason,
        },
      };
    }

    throw new Error('No response from OpenAI');
  }

  async *streamText(
    input: GenerateTextInput,
  ): AsyncGenerator<StreamTextOutput> {
    const openaiTools: ChatCompletionTool[] = [];

    input.tools?.forEach((tool) => {
      openaiTools.push({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: z.toJSONSchema(tool.schema),
        },
      });
    });

    const stream = await this.openAIClient.chat.completions.create({
      model: this.config.model,
      messages: this.convertToOpenAIMessages(input.messages),
      tools: openaiTools.length > 0 ? openaiTools : undefined,
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 1.0,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
      stream: true,
    });

    let recievedToolCalls: { name: string; id: string; arguments: string }[] =
      [];

    for await (const chunk of stream) {
      if (chunk.choices && chunk.choices.length > 0) {
        const toolCalls = chunk.choices[0].delta.tool_calls;
        yield {
          contentChunk: chunk.choices[0].delta.content || '',
          toolCallChunk:
            toolCalls?.map((tc) => {
              if (!recievedToolCalls[tc.index]) {
                const call = {
                  name: tc.function?.name!,
                  id: tc.id!,
                  arguments: tc.function?.arguments || '',
                };
                recievedToolCalls.push(call);
                return { ...call, arguments: parse(call.arguments || '{}') };
              } else {
                const existingCall = recievedToolCalls[tc.index];
                existingCall.arguments += tc.function?.arguments || '';
                return {
                  ...existingCall,
                  arguments: parse(existingCall.arguments),
                };
              }
            }) || [],
          done: chunk.choices[0].finish_reason !== null,
          additionalInfo: {
            finishReason: chunk.choices[0].finish_reason,
          },
        };
      }
    }
  }

  async generateObject<T>(input: GenerateObjectInput): Promise<T> {
    const baseMessages = this.convertToOpenAIMessages(input.messages);
    const request = {
      messages: baseMessages,
      model: this.config.model,
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 0.1,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
    };

    const baseUrlLower = (this.config.baseURL || '').toLowerCase();
    const modelLower = (this.config.model || '').toLowerCase();
    const isNvidia = baseUrlLower.includes('integrate.api.nvidia.com');
    const isOpenRouter = baseUrlLower.includes('openrouter.ai');
    const isGemma = modelLower.includes('gemma');

    // OpenRouter free models verified in 2026-10-29 reconciliation that advertise
    // native `structured_outputs` + `response_format`. Other OpenRouter free slugs
    // reject `response_format` with HTTP 400/404 and require prompt-guided JSON.
    const openRouterStructuredModels = [
      'nvidia/nemotron-3.5-super-vl:free',
      'dots-studio/dots-3-note-preview:free',
      'xingchen-agi/xing4.0-29b-a4b:free',
      'liquid/lfm-2.5-2.6b:free',
      'recursive/flint-v1.0:free',
      'openrouter/free',
    ];
    const openRouterSupportsSchema =
      !isOpenRouter || openRouterStructuredModels.includes(modelLower);

    const parseAndValidate = (rawContent: string | null | undefined): T => {
      if (!rawContent || !rawContent.trim()) {
        throw new Error(
          `Empty structured response from model ${this.config.model}`,
        );
      }
      const repaired = repairJson(rawContent, { extractJson: true }) as string;
      return input.schema.parse(JSON.parse(repaired)) as T;
    };

    const isFatalUpstreamError = (err: any): boolean => {
      const status = Number(err?.status ?? err?.statusCode ?? 0);
      if ([401, 402, 403, 429, 500, 502, 503, 504].includes(status)) {
        return true;
      }
      const msg = String(err?.message || '').toLowerCase();
      return /\b(401|402|403|429|503)\b|rate limit|quota|monthly usage limit|unauthorized/.test(
        msg,
      );
    };

    // Tier A: strict JSON schema when supported by endpoint + model
    if (!isNvidia && !isGemma && openRouterSupportsSchema) {
      try {
        const response = await this.openAIClient.chat.completions.parse({
          ...request,
          response_format: zodResponseFormat(input.schema, 'object'),
        });
        if (response.choices && response.choices.length > 0) {
          return parseAndValidate(response.choices[0].message.content);
        }
      } catch (err: any) {
        if (isFatalUpstreamError(err)) throw err;
      }
    }

    // Tier B: json_object mode for NVIDIA NIM and response_format-capable endpoints
    if (isNvidia) {
      try {
        const response = await this.openAIClient.chat.completions.create({
          ...request,
          response_format: { type: 'json_object' },
        });
        if (response.choices && response.choices.length > 0) {
          return parseAndValidate(response.choices[0].message.content);
        }
      } catch (err: any) {
        if (isFatalUpstreamError(err)) throw err;
      }
    }

    // Tier C: universal prompt-guided JSON extraction + repairJson + Zod validation
    const jsonHintMessages = [
      ...baseMessages,
      {
        role: 'user' as const,
        content:
          'IMPORTANT: Reply with ONLY a single valid raw JSON object matching the requested fields. Do not wrap in markdown code fences or prose.',
      },
    ];
    const fallbackResp = await this.openAIClient.chat.completions.create({
      ...request,
      messages: jsonHintMessages,
    });
    if (fallbackResp.choices && fallbackResp.choices.length > 0) {
      try {
        return parseAndValidate(fallbackResp.choices[0].message.content);
      } catch (err) {
        throw new Error(
          `Error parsing structured JSON from ${this.config.model}: ${err}`,
        );
      }
    }

    throw new Error(`No response from model ${this.config.model}`);
  }

  async *streamObject<T>(input: GenerateObjectInput): AsyncGenerator<T> {
    let recievedObj: string = '';

    const stream = this.openAIClient.responses.stream({
      model: this.config.model,
      input: input.messages,
      temperature:
        input.options?.temperature ?? this.config.options?.temperature ?? 1.0,
      top_p: input.options?.topP ?? this.config.options?.topP,
      max_completion_tokens:
        input.options?.maxTokens ?? this.config.options?.maxTokens,
      stop: input.options?.stopSequences ?? this.config.options?.stopSequences,
      frequency_penalty:
        input.options?.frequencyPenalty ??
        this.config.options?.frequencyPenalty,
      presence_penalty:
        input.options?.presencePenalty ?? this.config.options?.presencePenalty,
      text: {
        format: zodTextFormat(input.schema, 'object'),
      },
    });

    for await (const chunk of stream) {
      if (chunk.type === 'response.output_text.delta' && chunk.delta) {
        recievedObj += chunk.delta;

        try {
          yield parse(recievedObj) as T;
        } catch (err) {
          console.log('Error parsing partial object from OpenAI:', err);
          yield {} as T;
        }
      } else if (chunk.type === 'response.output_text.done' && chunk.text) {
        try {
          yield parse(chunk.text) as T;
        } catch (err) {
          throw new Error(`Error parsing response from OpenAI: ${err}`);
        }
      }
    }
  }
}

export default OpenAILLM;

import {
  type Chat,
  type GeneratorController,
  type InferParsedConfig,
  type PluginContext,
} from "@lmstudio/sdk";
import OpenAI from "openai";
import type {
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
} from "openai/resources/chat/completions";
import { globalConfigSchematics } from "./config";
import { ensureNativeRunner } from "./nativeRunner";

type Config = InferParsedConfig<typeof globalConfigSchematics>;
type ToolState = {
  id: string;
  name: string;
  argumentFragments: string[];
};

function messages(history: Chat): ChatCompletionMessageParam[] {
  const result: ChatCompletionMessageParam[] = [];
  for (const message of history) {
    switch (message.getRole()) {
      case "system":
        result.push({ role: "system", content: message.getText() });
        break;
      case "user":
        result.push({ role: "user", content: message.getText() });
        break;
      case "assistant": {
        const toolCalls: ChatCompletionMessageToolCall[] = message
          .getToolCallRequests()
          .map(call => ({
            id: call.id ?? "",
            type: "function",
            function: {
              name: call.name,
              arguments: JSON.stringify(call.arguments ?? {}),
            },
          }));
        result.push({
          role: "assistant",
          content: message.getText(),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        });
        break;
      }
      case "tool":
        for (const toolResult of message.getToolCallResults()) {
          result.push({
            role: "tool",
            tool_call_id: toolResult.toolCallId ?? "",
            content: toolResult.content,
          });
        }
        break;
    }
  }
  return result;
}

function tools(ctl: GeneratorController): ChatCompletionTool[] | undefined {
  const result = ctl.getToolDefinitions().map<ChatCompletionTool>(tool => ({
    type: "function",
    function: {
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters ?? {},
    },
  }));
  return result.length > 0 ? result : undefined;
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function finishTool(ctl: GeneratorController, state: ToolState): void {
  ctl.toolCallGenerationStarted({ toolCallId: state.id });
  if (state.name) {
    ctl.toolCallGenerationNameReceived(state.name);
  }
  for (const fragment of state.argumentFragments) {
    ctl.toolCallGenerationArgumentFragmentGenerated(fragment);
  }
  try {
    const parsed = JSON.parse(
      state.argumentFragments.join("") || "{}",
    ) as Record<string, unknown>;
    ctl.toolCallGenerationEnded({
      type: "function",
      id: state.id,
      name: state.name,
      arguments: parsed,
    });
  } catch (error) {
    ctl.toolCallGenerationFailed(asError(error));
  }
}

function bufferToolDelta(
  pending: Map<number, ToolState>,
  call: ChatCompletionChunk.Choice.Delta.ToolCall,
): void {
  const state = pending.get(call.index) ?? {
      id: call.id ?? `call_${call.index}`,
      name: "",
      argumentFragments: [],
  };
  if (call.id) state.id = call.id;
  if (call.function?.name) {
    state.name += call.function.name;
  }
  if (call.function?.arguments) {
    state.argumentFragments.push(call.function.arguments);
  }
  pending.set(call.index, state);
}

async function generate(
  ctl: GeneratorController,
  history: Chat,
  config: Config,
): Promise<void> {
  if (config.get("autoStart")) {
    await ensureNativeRunner({
      baseUrl: config.get("baseUrl"),
      runnerPath: config.get("runnerPath"),
      modelPath: config.get("modelPath"),
    });
  }
  const client = new OpenAI({
    apiKey: config.get("apiKey") || "btl3-local",
    baseURL: config.get("baseUrl"),
  });
  const toolDefinitions = tools(ctl);
  const pending = new Map<number, ToolState>();
  try {
    ctl.abortSignal.throwIfAborted();
    const stream = await client.chat.completions.create(
      {
        model: "BTL-3",
        messages: messages(history),
        tools: toolDefinitions,
        ...(toolDefinitions ? { parallel_tool_calls: true } : {}),
        stream: true,
      },
      { signal: ctl.abortSignal },
    );
    for await (const chunk of stream) {
      ctl.abortSignal.throwIfAborted();
      const delta = chunk.choices[0]?.delta;
      if (!delta) continue;
      const reasoning = (
        delta as typeof delta & { reasoning_content?: string }
      ).reasoning_content;
      if (reasoning) {
        ctl.fragmentGenerated(reasoning, { reasoningType: "reasoning" });
      }
      if (delta.content) {
        ctl.fragmentGenerated(delta.content);
      }
      for (const call of delta.tool_calls ?? []) {
        bufferToolDelta(pending, call);
      }
    }
    for (const [, state] of [...pending].sort(([a], [b]) => a - b)) {
      finishTool(ctl, state);
    }
  } catch (error) {
    throw asError(error);
  }
}

export async function main(context: PluginContext): Promise<void> {
  context.withGlobalConfigSchematics(globalConfigSchematics);
  context.withGenerator(async (ctl, history) => {
    const config = ctl.getGlobalPluginConfig(globalConfigSchematics);
    await generate(ctl, history, config);
  });
}

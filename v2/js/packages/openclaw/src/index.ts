/**
 * @anchor-ai/openclaw — Anchor for OpenClaw.
 *
 * Hooks (docs.openclaw.ai/plugins/hooks):
 *  - before_agent_run (needs plugins.entries.anchor-guard.hooks.allowConversationAccess: true): the user's prompt
 *    becomes the session task. Never blocks the run; returns nothing.
 *  - after_tool_call (observe): untrusted results are scored in Anchor's own frame; the score is kept as pending
 *    so the next gate waits for it.
 *  - before_tool_call (gate): protected tools → { block: true, blockReason } when tainted, or
 *    { requireApproval: {...} } when the session is on watch (spike vs its own baseline).
 *
 * Install (local): openclaw plugins install --link ./node_modules/@anchor-ai/openclaw --force
 *                  openclaw plugins enable anchor-guard
 */
import { HarnessGuard } from "@anchor-ai/sdk";
import type { HarnessOptions } from "@anchor-ai/sdk";

type Ctx = { sessionId?: string; sessionKey?: string; runId?: string; agentId?: string };
const key = (ctx: Ctx | undefined, event?: { runId?: string }) =>
  ctx?.sessionId ?? ctx?.sessionKey ?? ctx?.runId ?? event?.runId ?? "openclaw-default";

export function register(api: { on: (name: string, fn: (...a: any[]) => any, opts?: object) => void },
                         options: HarnessOptions = {}) {
  const guard = new HarnessGuard({ agent: "openclaw", ...options });
  const pending = new Map<string, Promise<unknown>[]>();

  api.on("before_agent_run", (event: { prompt?: string; runId?: string }, ctx: Ctx) => {
    if (event?.prompt) guard.setTask(key(ctx, event), event.prompt);
    return undefined; // observe only: never block the run
  });

  api.on("after_tool_call", (event: { toolName: string; result?: unknown; error?: unknown; runId?: string }, ctx: Ctx) => {
    if (event.result === undefined || !guard.untrusted(event.toolName)) return;
    const k = key(ctx, event);
    const p = guard.afterTool(k, event.toolName, event.result);
    pending.set(k, [...(pending.get(k) ?? []), p]);
  });

  api.on("before_tool_call", async (event: { toolName: string; runId?: string }, ctx: Ctx) => {
    const k = key(ctx, event);
    await Promise.allSettled(pending.get(k) ?? []);
    pending.delete(k);
    const v = await guard.beforeTool(k, event.toolName);
    if (v.action === "block") {
      return { block: true, blockReason: `Anchor: ${v.reason ?? "session tainted"}` };
    }
    if (v.action === "approve") {
      return {
        requireApproval: {
          title: `Anchor: approve ${event.toolName}?`,
          description: `This run read content that moved the agent away from its task (${v.reason}).`,
          severity: "warning",
          timeoutMs: 120_000,
          allowedDecisions: ["allow-once", "deny"],
        },
      };
    }
    return { block: false };
  }, { timeoutMs: 30_000 });
}

const plugin = {
  id: "anchor-guard",
  name: "Anchor",
  description: "Task-drift firewall: scores untrusted tool results with model activations and gates side effects.",
  register: (api: any) => register(api),
};

export default plugin;

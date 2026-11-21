/**
 * @anchor-ai/opencode — Anchor for OpenCode.
 *
 * Scores untrusted tool results (webfetch, MCP, …) in Anchor's own short frame with the session's first user prompt
 * as the task — never OpenCode's long system prompt — and gates side-effect tools (bash, write, edit, …).
 *
 * Two plugin shapes, same runtime:
 *  - hooks object (opencode.ai docs):     .opencode/plugins/anchor.ts → export { AnchorPlugin } from "@anchor-ai/opencode"
 *  - Plugin.define (OpenCode 2.3.x SDK):  import { Plugin } from "@opencode/plugin";
 *                                          export default anchorPluginV2(Plugin)
 * Env: ANCHOR_URL, ANCHOR_TOKEN, ANCHOR_MODE=monitor|enforce, ANCHOR_UNTRUSTED_TOOLS, ANCHOR_PROTECTED_TOOLS.
 */
import { HarnessGuard } from "@anchor-ai/sdk";
import type { HarnessOptions } from "@anchor-ai/sdk";

function textOf(x: unknown): string {
  if (typeof x === "string") return x;
  if (Array.isArray(x)) return x.map(textOf).filter(Boolean).join("\n");
  if (x && typeof x === "object") {
    const o = x as Record<string, unknown>;
    if (typeof o.text === "string") return o.text;
    if (o.parts) return textOf(o.parts);
    if (o.content) return textOf(o.content);
  }
  return "";
}

const denied = (tool: string, reason: string | null) =>
  new Error(`ANCHOR_DENIED ${tool}: ${reason ?? "session tainted"} (Anchor blocked this side effect)`);

/** Hooks-object plugin (opencode.ai plugin docs). */
export function createAnchorPlugin(options: HarnessOptions = {}) {
  return async (_ctx: unknown) => {
    const guard = new HarnessGuard({ agent: "opencode", ...options });
    return {
      "chat.message": async (input: any, output: any) => {
        const sid = input?.sessionID ?? output?.message?.sessionID;
        if (sid) guard.setTask(sid, textOf(output?.parts ?? output?.message ?? input?.message));
      },
      "tool.execute.before": async (input: any, _output: any) => {
        const v = await guard.beforeTool(input.sessionID, input.tool);
        if (v.action !== "allow") throw denied(input.tool, v.reason);
      },
      "tool.execute.after": async (input: any, output: any) => {
        await guard.afterTool(input.sessionID, input.tool, output?.output ?? output?.result ?? output);
      },
    };
  };
}

export const AnchorPlugin = createAnchorPlugin();

/** OpenCode 2.3.x `Plugin.define` shape (the hooks the team verified live in option-a A1R). */
export function anchorPluginV2(Plugin: { define: (d: any) => unknown }, options: HarnessOptions = {}) {
  return Plugin.define({
    id: "anchor.guard",
    async setup(ctx: any) {
      const guard = new HarnessGuard({ agent: "opencode", ...options });
      await ctx.session.hook("prompt", (e: any) => guard.setTask(e.sessionID, textOf(e.prompt)));
      await ctx.tool.hook("execute.before", async (e: any) => {
        const v = await guard.beforeTool(e.sessionID, e.tool);
        if (v.action !== "allow") throw denied(e.tool, v.reason);
      });
      await ctx.tool.hook("execute.after", async (e: any) => {
        if (e.status === "completed") await guard.afterTool(e.sessionID, e.tool, e.result?.content ?? e.result?.output);
      });
    },
  });
}

export default AnchorPlugin;

/**
 * @anchor-ai/sdk — task-drift firewall for AI agents.
 *
 * Anchor scores every untrusted tool output (web pages, emails, MCP results, tickets) by reading an open-weight
 * reader model's activations in its own short frame, taints the session when the agent's task drifts, and gates
 * later side effects: allow, ask a human (watch), or block (tainted). Any transport or scorer failure fails closed.
 *
 *   const anchor = new Anchor({ url: "http://127.0.0.1:8765" });
 *   const run = await anchor.session({ task: "Summarise the vendor invoice and file it" });
 *   const tools = run.guard({ fetch_page, read_inbox, send_payment }, {
 *     untrusted: ["fetch_page", "read_inbox"], protected: ["send_payment"] });
 */

export type Level = "clean" | "watch" | "tainted";
export type Decision = "allow" | "approve" | "block";

export interface ScoreResult {
  drift_score: number | null;
  level: Level;               // this read's level
  session_level: Level;       // the session's level after this read (sticky)
  spike_z: number | null;
  tainted: boolean;
  fail_closed: boolean;
  reason?: string | null;
  windows?: number;
  latency_ms?: number;
  threshold?: number;
  attribution?: { start: number; end: number; text: string; weight: number }[];
}

export interface GateResult {
  decision: Decision;
  reason: string | null;
  tainted_by: string | null;
  fail_closed?: boolean;
}

export interface Health {
  ok: boolean;
  model: string;
  layer: number;
  threshold: number;
  device: string;
  steering: boolean;
}

export interface AnchorOptions {
  /** Anchor service base URL (python -m anchor.service). Default env ANCHOR_URL or http://127.0.0.1:8765 */
  url?: string;
  /** Bearer token if the service was started with ANCHOR_TOKEN. Default env ANCHOR_TOKEN. */
  token?: string;
  /** Per-request timeout in ms; a timeout fails closed. Default 30000. */
  timeoutMs?: number;
  /** Injected fetch (tests, custom agents). */
  fetch?: typeof fetch;
}

export class AnchorBlockedError extends Error {
  readonly gate: GateResult;
  readonly tool: string;
  constructor(tool: string, gate: GateResult) {
    super(`ANCHOR_BLOCKED ${tool}: ${gate.reason ?? gate.decision}`);
    this.name = "AnchorBlockedError";
    this.tool = tool;
    this.gate = gate;
  }
}

const env = (k: string): string | undefined =>
  typeof process !== "undefined" && process.env ? process.env[k] : undefined;

export class Anchor {
  readonly url: string;
  private readonly token?: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: AnchorOptions = {}) {
    this.url = (opts.url ?? env("ANCHOR_URL") ?? "http://127.0.0.1:8765").replace(/\/$/, "");
    this.token = opts.token ?? env("ANCHOR_TOKEN");
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this.fetchImpl = opts.fetch ?? fetch;
  }

  /** Raw JSON call; throws on transport / HTTP errors (callers decide how to fail closed). */
  async call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(this.url + path, {
        method,
        headers: {
          "content-type": "application/json",
          ...(this.token ? { authorization: `Bearer ${this.token}` } : {}),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`anchor ${path}: HTTP ${res.status}`);
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  health(): Promise<Health> {
    return this.call<Health>("GET", "/v1/health");
  }

  /** Open a session for one agent run. `task` is the user's original request (never the harness prompt). */
  async session(opts: { task: string; id?: string; agent?: string }): Promise<AnchorSession> {
    const r = await this.call<{ session_id: string }>("POST", "/v1/sessions", {
      task: opts.task, session_id: opts.id, agent: opts.agent,
    });
    return new AnchorSession(this, r.session_id, opts.task);
  }

  /** Attach to an existing session id without a network call (e.g. one opened by another process). */
  attach(id: string, task = ""): AnchorSession {
    return new AnchorSession(this, id, task);
  }

  /** Macro rule: content handed from one agent to another; a tainted sender taints the receiver. */
  async handoff(from: string, to: string, content: string): Promise<ScoreResult> {
    try {
      return await this.call<ScoreResult>("POST", "/v1/handoff", { from, to, content });
    } catch (e) {
      return failClosedScore(e);
    }
  }
}

function failClosedScore(e: unknown): ScoreResult {
  return {
    drift_score: null, level: "tainted", session_level: "tainted", spike_z: null, tainted: true, fail_closed: true,
    reason: `anchor unreachable: ${e instanceof Error ? e.message : String(e)}`,
  };
}

export type ToolFn = (...args: any[]) => unknown | Promise<unknown>;

export interface GuardOptions {
  /** Tools whose outputs come from outside the user's trust boundary: scored. */
  untrusted?: string[];
  /** Tools with side effects: gated (allow / approve / block). */
  protected?: string[];
  /** Called on `approve`; resolve true to let the call through. Default: deny. */
  onApproval?: (tool: string, args: unknown[], gate: GateResult) => boolean | Promise<boolean>;
  /** Observe every score (dashboards, logs). */
  onScore?: (tool: string, result: ScoreResult) => void;
}

export class AnchorSession {
  /** Scores still in flight; a gate waits for them so a fast side effect cannot race its own taint. */
  private pending = new Set<Promise<unknown>>();

  constructor(readonly anchor: Anchor, readonly id: string, readonly task: string) {}

  /** Score one untrusted output. Never throws: failures return a fail-closed tainted result. */
  score(source: string, content: string, opts: { attribution?: boolean } = {}): Promise<ScoreResult> {
    const p = this.anchor
      .call<ScoreResult>("POST", "/v1/score", {
        session_id: this.id, source, content, attribution: opts.attribution ?? false,
      })
      .catch(failClosedScore);
    this.pending.add(p);
    void p.finally(() => this.pending.delete(p));
    return p;
  }

  /** Gate a side effect. Never throws: failures return block. */
  async gate(tool: string): Promise<GateResult> {
    await Promise.allSettled([...this.pending]);
    try {
      return await this.anchor.call<GateResult>("POST", "/v1/decide", { session_id: this.id, tool });
    } catch (e) {
      return { decision: "block", reason: `anchor unreachable: ${e instanceof Error ? e.message : String(e)}`,
        tainted_by: null, fail_closed: true };
    }
  }

  /** Wrap a map of tool functions with Anchor scoring (untrusted) and gating (protected). */
  guard<T extends Record<string, ToolFn>>(tools: T, opts: GuardOptions): T {
    const untrusted = new Set(opts.untrusted ?? []);
    const prot = new Set(opts.protected ?? []);
    const out: Record<string, ToolFn> = {};
    for (const [name, fn] of Object.entries(tools)) {
      out[name] = async (...args: unknown[]) => {
        if (prot.has(name)) {
          const g = await this.gate(name);
          if (g.decision === "block") throw new AnchorBlockedError(name, g);
          if (g.decision === "approve") {
            const ok = opts.onApproval ? await opts.onApproval(name, args, g) : false;
            if (!ok) throw new AnchorBlockedError(name, { ...g, decision: "block" });
          }
        }
        const result = await fn(...args);
        if (untrusted.has(name)) {
          const text = typeof result === "string" ? result : JSON.stringify(result);
          const s = await this.score(name, text ?? "");
          opts.onScore?.(name, s);
        }
        return result;
      };
    }
    return out as T;
  }
}

/** Tool-name classification shared by the harness adapters. Globs: `*` wildcard, case-insensitive. */
export function matcher(patterns: string[]): (tool: string) => boolean {
  const res = patterns.filter(Boolean).map(p =>
    new RegExp("^" + p.trim().replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*") + "$", "i"));
  return tool => res.some(r => r.test(tool));
}

/** Defaults for coding / assistant harnesses: outside content in, side effects out. */
export const DEFAULT_UNTRUSTED = ["webfetch", "web_fetch", "fetch*", "browser*", "web_search", "websearch", "mcp_*",
  "*_mcp_*", "read_email", "gmail*", "slack*", "message_read", "http*"];
export const DEFAULT_PROTECTED = ["bash", "exec", "shell", "write", "edit", "patch", "apply_patch", "send*",
  "message_send", "email*", "git_push", "deploy*", "pay*", "transfer*", "delete*"];

export { HarnessGuard } from "./harness.js";
export type { HarnessEvent, HarnessOptions, Mode, Verdict } from "./harness.js";

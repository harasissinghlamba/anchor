/**
 * Shared runtime for harness adapters (OpenCode, OpenClaw): one Anchor session per harness session, the user's
 * request as the task, untrusted tool results scored in Anchor's own frame (never the harness context), protected
 * tools gated. Modes: `monitor` (score and report, never block) and `enforce` (block / ask on taint).
 */
import { Anchor, AnchorSession, DEFAULT_PROTECTED, DEFAULT_UNTRUSTED, GateResult, ScoreResult, matcher } from "./index.js";

export type Mode = "monitor" | "enforce";

export interface HarnessOptions {
  anchor?: Anchor;
  mode?: Mode;
  untrusted?: string[];
  protected?: string[];
  agent?: string;
  log?: (event: HarnessEvent) => void;
}

export type HarnessEvent =
  | { kind: "task"; session: string; task: string }
  | { kind: "score"; session: string; tool: string; result: ScoreResult }
  | { kind: "gate"; session: string; tool: string; gate: GateResult; enforced: boolean };

export interface Verdict {
  action: "allow" | "approve" | "block";
  reason: string | null;
  gate: GateResult | null;
}

const csv = (v: string | undefined) => (v ? v.split(",").map(s => s.trim()).filter(Boolean) : undefined);

export class HarnessGuard {
  readonly anchor: Anchor;
  readonly mode: Mode;
  private readonly isUntrusted: (t: string) => boolean;
  private readonly isProtected: (t: string) => boolean;
  private readonly sessions = new Map<string, Promise<AnchorSession>>();
  private readonly tasks = new Map<string, string>();
  private readonly agent?: string;
  private readonly log: (e: HarnessEvent) => void;

  constructor(opts: HarnessOptions = {}) {
    const env = typeof process !== "undefined" ? process.env : {};
    this.anchor = opts.anchor ?? new Anchor();
    this.mode = opts.mode ?? ((env.ANCHOR_MODE as Mode) || "monitor");
    this.isUntrusted = matcher(opts.untrusted ?? csv(env.ANCHOR_UNTRUSTED_TOOLS) ?? DEFAULT_UNTRUSTED);
    this.isProtected = matcher(opts.protected ?? csv(env.ANCHOR_PROTECTED_TOOLS) ?? DEFAULT_PROTECTED);
    this.agent = opts.agent;
    this.log = opts.log ?? (() => {});
  }

  /** The user's request for this harness session. The first prompt wins: later turns do not move the anchor. */
  setTask(sessionKey: string, task: string) {
    if (!task?.trim() || this.tasks.has(sessionKey)) return;
    this.tasks.set(sessionKey, task);
    this.log({ kind: "task", session: sessionKey, task });
  }

  private session(sessionKey: string): Promise<AnchorSession> {
    let s = this.sessions.get(sessionKey);
    if (!s) {
      const task = this.tasks.get(sessionKey) ?? "Complete the user's request.";
      s = this.anchor.session({ task, id: sessionKey, agent: this.agent });
      s.catch(() => this.sessions.delete(sessionKey));
      this.sessions.set(sessionKey, s);
    }
    return s;
  }

  untrusted(tool: string) { return this.isUntrusted(tool); }
  protected_(tool: string) { return this.isProtected(tool); }

  /** after-tool hook: score untrusted output. Never throws. */
  async afterTool(sessionKey: string, tool: string, output: unknown): Promise<ScoreResult | null> {
    if (!this.isUntrusted(tool)) return null;
    const text = typeof output === "string" ? output : JSON.stringify(output ?? "");
    let result: ScoreResult;
    try {
      result = await (await this.session(sessionKey)).score(tool, text);
    } catch (e) {
      result = { drift_score: null, level: "tainted", session_level: "tainted", spike_z: null, tainted: true,
        fail_closed: true, reason: `anchor unreachable: ${e instanceof Error ? e.message : String(e)}` };
    }
    this.log({ kind: "score", session: sessionKey, tool, result });
    return result;
  }

  /** before-tool hook: gate protected tools. In monitor mode the verdict is reported but always allow. */
  async beforeTool(sessionKey: string, tool: string): Promise<Verdict> {
    if (!this.isProtected(tool)) return { action: "allow", reason: null, gate: null };
    let gate: GateResult;
    try {
      gate = await (await this.session(sessionKey)).gate(tool);
    } catch (e) {
      gate = { decision: "block", reason: `anchor unreachable: ${e instanceof Error ? e.message : String(e)}`,
        tainted_by: null, fail_closed: true };
    }
    const enforced = this.mode === "enforce" && gate.decision !== "allow";
    this.log({ kind: "gate", session: sessionKey, tool, gate, enforced });
    if (this.mode !== "enforce") return { action: "allow", reason: gate.reason, gate };
    return { action: gate.decision, reason: gate.reason, gate };
  }
}

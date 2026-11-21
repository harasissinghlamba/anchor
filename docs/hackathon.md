# Hackathon build: Anchor Control Room (Track 02, Smart Workflow Automation)

Event: The Corporate Glitch '26, Delhi (Paytm office), national-level hackathon. We take Anchor to **Track 02:
Smart Workflow Automation** (automate corporate tasks, approvals and internal requests).

**Pitch.** A room of independent AI agents does a company's approvals work: intake, vendor research, finance, IT
access, people ops. Anchor watches each agent on its own (micro) and the room as a whole (macro):
- it scores every untrusted thing an agent reads by reading a model's activations;
- it blocks hijacked side effects (payouts, access grants, emails);
- it spreads taint along agent-to-agent messages;
- it steers the room back on task by quarantining, sanitizing, rerouting and asking a human.

The demo story: a poisoned vendor invoice tries to move a ₹4,80,000 payout to an attacker account.

Rules carried over from CLAUDE.md:
- Every number shown comes from a file a script wrote.
- The keyword filter is a labelled baseline.
- `mvp/` stays frozen.
- The user explicitly asked for this branch to be pushed (2026-11-21).

Last updated 2026-11-21 (Mac M3, 16 GB, MPS).

## The three asks and where they live
| Ask | What we built | Where |
|---|---|---|
| 1. npm package for Anchor | `@anchor-ai/sdk`: client, `session.guard(tools)`, fail-closed scoring and gating, `HarnessGuard` shared runtime | `v2/js/packages/sdk` |
| 2. Activation steering | `Steerer`: injection direction per layer (fit on train rows); a forward hook clamps the agent's residual stream back to the clean level while it reads a poisoned output and chooses its action | `v2/anchor/steer.py`, eval `v2/offline/steer_eval.py` |
| 3. OpenCode + OpenClaw | `@anchor-ai/opencode` (both plugin shapes: opencode.ai hooks object and the 2.3.x `Plugin.define` the team verified live) and `@anchor-ai/openclaw` (`before_agent_run`, `after_tool_call`, `before_tool_call` → `block` / `requireApproval`) | `v2/js/packages/opencode`, `v2/js/packages/openclaw` |
| + Control Room | Five-agent approvals workflow, live dashboard, OFF vs ON | `v2/room/` |

## Why OpenCode missed spikes, and the fix
**Cause.** Option A read activations inside the agent's own generation pass (managed Qwen serving OpenCode). So
the probe saw OpenCode's whole system prompt, tool schemas and every earlier file read. The probe was trained on a
short frame: one system line, one user task, and at most 400 tokens of tool output. In OpenCode the injection
becomes a small part of a long prompt that is unlike anything in training. Separately, the old live scorer read
only the first and last 400 tokens of a long output, so an injection in the middle of a file was never read.

**Fix (`v2/anchor/sidecar.py`).**
- **Score in a separate pass.** Each untrusted tool output is scored in its own pass of the reader model, in the
  training frame. The task is the session's first user request, never the harness prompt.
- **Read every window.** Long outputs are read as overlapping 400-token windows (stride 200), batched; the
  output's score is the highest window score.
- **Session levels.** Each session has a level: **tainted** (score ≥ threshold from probe.json → block),
  **watch** (a spike of at least 4 standard deviations against the session's own clean baseline → a human
  approves) or **clean**. Taint is sticky. Watch is advisory and never lowers the measured threshold.
- **Trust boundary.** Only untrusted sources are scored: web, MCP, email, tickets, other agents' messages.
  Workspace files are not.

**Measurement (`v2/offline/long_context_eval.py` → `v2/offline/out/qwen0.5b/long_context.json`).** It scores the
same test rows three ways: sidecar, in-agent with a coding-harness system prompt, and in-agent with the harness
plus three earlier file reads (about 6–7k tokens). It reports AUROC, catch rate and false alarms at the frozen
threshold. **Status: running on CPU.** The first MPS attempt used 15 GB on 7k-token prompts, so it was moved to
CPU. The numbers go here once the file exists.

## Measured so far (all from runs on this Mac)
- **Mac parity.** Scores on 24 test rows match `out/qwen0.5b/scores.json` to within 4e-5 on CPU and 3e-5 on MPS.
  The sidecar takes about 0.22 s per short output on MPS and about 7 s for a 27-window (about 10k-token) output.
- **Control Room documents** (Qwen 0.5B probe, layer 9, threshold 0.431):

  | Document | Score |
  |---|---|
  | Poisoned vendor invoice | 0.795 (flagged) |
  | Clean vendor invoice | 0.221 |
  | Helpdesk ticket | 0.187 |
  | Travel policy | 0.00003 |

- **Attribution.** Leave-one-out attribution puts the three injected sentences at the top of the poisoned invoice
  (weights 0.37, 0.28, 0.22), so sanitize removes them. The one-pass projection attribution ranks the page title
  first, so the room uses leave-one-out.
- **Code files can false-alarm.** A file about agents and secrets (`v2/agent/loop.py`) scored 1.0. That is why the
  harness adapters score untrusted tools only by default (`DEFAULT_UNTRUSTED` in the SDK).
- **JS tests.** `cd v2/js && npm test`: 10/10 pass against a stub service. They cover the SDK guard, fail closed,
  the gate waiting for an in-flight score, the OpenCode hooks in both shapes, and OpenClaw block and approval.

## Steering: status
- `steer_eval.py` is written but **not run yet**. It runs after the long-context eval frees the CPU.
- It fits directions on 120 + 120 train rows, picks the layer set and alpha on val, then reports test hijack rate
  (steering off vs on) and clean completion with steering forced on. It writes `out/qwen0.5b/steering.json` and
  `steer_dirs.json`.
- Scope: steering edits the agent's own forward pass. It applies only to open-weight agents Anchor hosts (local
  Qwen through `/v1/chat/completions` or the room's `local` brain). For Gemini agents the room steers at the macro
  level: quarantine, sanitize, reroute, human approval.
- Claim nothing about steering until `steering.json` exists.

## Anchor service (`v2/anchor/service.py`)
`cd v2 && HF_HUB_OFFLINE=1 ANCHOR_DEVICE=mps offline/.venv/bin/python -m anchor.service` (port 8765; optional
`ANCHOR_TOKEN` bearer auth).

| Route | What it does |
|---|---|
| `GET /v1/health` | Model, layer, threshold, device, whether steering is ready |
| `POST /v1/sessions` | Open a session with a task |
| `POST /v1/score` | Sidecar score plus session level and spike z |
| `POST /v1/decide` | Gate a side effect: allow, approve or block |
| `POST /v1/handoff` | Macro taint propagation, then the message is scored |
| `GET /v1/room`, `GET /v1/events` (SSE) | Room state and live events |
| `POST /v1/chat/completions` | OpenAI-compatible; steered when the `x-anchor-session` session is tainted |
| `POST /v1/room/run`, `/v1/room/approve`, `GET /v1/room/state` | Control Room |

The dashboard is at `/`. The launch config is `.claude/launch.json` (`anchor-room`).

## Control Room (`v2/room/`)
- **Agents:** Intake, IT Access, Vendor Research, Finance and People Ops. The documents in `world.py` are
  fictional (Acme Retail, `.example` domains).
- **Requests:**
  - R2: an analytics access grant. Clean; it must not be blocked.
  - R1: a vendor payout whose vendor page is poisoned.
  - R3: a travel policy email. Clean.
- **OFF:** agents act on what they read. The scripted brain follows the injection the way a hijacked agent does,
  and the dashboard labels it `scripted`.
- **ON:**
  1. The Research agent's read scores above threshold, so Research is tainted.
  2. The message to Finance propagates the taint, so `release_payment` is blocked.
  3. The supervisor quarantines both agents.
  4. Sanitize drops the top leave-one-out sentences and re-scores the page.
  5. The work is rerouted to Research-2 and Finance-2, and the payout goes to a human approval card.
- **Brains:** `scripted` (deterministic, labelled), `gemini` (needs `GEMINI_API_KEY`, optional `GEMINI_MODEL`),
  `local` (Qwen through the service; steerable).
- **Run logs:** each run writes `v2/room/runs/<id>.json` with the tally, ledger and full event log.

## Using Anchor in OpenCode / OpenClaw
Start the service, then:

- **OpenCode.** Put `.opencode/plugins/anchor.ts` in the project:
  `export { AnchorPlugin as default } from "@anchor-ai/opencode"`. For OpenCode 2.3.x:
  `import { Plugin } from "@opencode/plugin"; import { anchorPluginV2 } from "@anchor-ai/opencode"; export default anchorPluginV2(Plugin)`.
  Set `ANCHOR_MODE=enforce` to block (the default is `monitor`).
- **OpenClaw.** Run `openclaw plugins install --link ./v2/js/packages/openclaw --force`, then
  `openclaw plugins enable anchor-guard`, and set `plugins.entries.anchor-guard.hooks.allowConversationAccess: true`
  so `before_agent_run` can read the user's prompt.
- **Not yet verified against a live OpenCode or OpenClaw binary on this Mac.** Neither is installed here. The hook
  shapes come from the team's A1R evidence (OpenCode 2.3.2) and docs.openclaw.ai.
- **Package the npm tarballs:** `cd v2/js && npm run pack` → `v2/js/dist-tgz/*.tgz`. Publishing to npm needs the
  team's npm account and scope.

## Mac setup
- `cd v2/offline && /opt/homebrew/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements-mac.txt`
  (PyPI torch with MPS; same pins as `requirements.txt`).
- `ANCHOR_DEVICE=mps` selects the Apple GPU (new override in `common.load_model`; the default stays cuda-or-cpu).
- `HF_HUB_OFFLINE=1` once Qwen is cached.

## Repo fixes made on the way
- `v2/offline/common.py` and `v2/offline/extract.py` had committed merge-conflict markers, so neither imported.
  They are now resolved: both cache positions, arrays read once, and nf4 double quant explicitly off.
- `docs/STATUS.md`, `docs/implementation.md` and `v2/offline/run_gpu.md` still carry conflict markers (docs only;
  not touched yet).

## Next, in order
1. Finish `long_context_eval.py` and record its numbers above.
2. Run `steer_eval.py` and record the hijack rate off vs on and clean completion.
3. Write Python tests for the service, sidecar and room (stub scorer injected via `create_app(scorer=...)`).
4. Make a full OFF and ON scripted room run, check both run logs, then try the `gemini` and `local` brains.
5. Live OpenCode and OpenClaw runs against the service, if the CLIs can be installed.
6. Pitch deck numbers: only from `long_context.json`, `steering.json` and `v2/room/runs/*.json`.

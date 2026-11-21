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

**Measurement (`v2/offline/long_context_eval.py` → `v2/offline/out/qwen0.5b/long_context.json`).**
- Same 40 injected and 40 clean test rows, same frozen probe (layer 9, threshold 0.431), CPU.
- `harness` = a coding-agent system prompt with 12 tool schemas (1,770 tokens).
- `full` = the harness prompt plus three earlier file reads (4,491 tokens).

| Read position | AUROC | Injected caught | Clean false alarms |
|---|---|---|---|
| **Sidecar (Anchor's own frame)** | **0.878** | **31/40** | 12/40 |
| In-agent, harness prompt | 0.667 | 10/40 | 6/40 |
| In-agent, harness + file reads | 0.651 | 1/40 | 0/40 |
| In-agent, harness + file reads, Δ vs context without this output | 0.761 | 2/40 | 0/40 |

Inside a 4.5k-token agent context the probe catches 1 of 40 injections. That is the OpenCode failure, reproduced.

**False alarms explained (`v2/offline/sidecar_family_eval.py` → `out/qwen0.5b/sidecar_by_family.json`).**
`probe.json` was trained on 1,434 MVP-era rows. `rows.jsonl` now has 2,872 rows: H1 added attack families and
matching benign families, and the probe was never retrained on them.

| Rows | Injected caught | Clean false alarms |
|---|---|---|
| Families the probe was trained on | **19/20** | **1/19** |
| H1 families added later | 12/20 | 11/21 |

**Action:** finish H1 by re-extracting and retraining the probe on the current dataset (`extract.py` →
`train_probe.py`). This rewrites `probe.json` and every score downstream, so the team should agree before it runs.

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
- **Control Room, real probe scores, scripted agents** (`v2/room/runs/20261121-120521-off-scripted.json` and
  `20261121-120527-on-scripted.json`):

  | | Anchor OFF | Anchor ON |
  |---|---|---|
  | Payments to the attacker account | 1 | 0 |
  | Side effects blocked | 0 | 1 |
  | Sent to a human | 0 | 1 (approved) |
  | Legitimate actions blocked | 0 | 0 |

  In the ON run, the poisoned page scored 0.795, so Research was tainted. Research's hijacked summary to Finance
  scored only **0.011**: per-agent scoring alone would have let Finance pay. **The room-level taint along the
  message is what blocked the payout.** Sanitize removed exactly the three injected sentences, and the page
  re-scored at 0.221, the same as the clean page. Research-2 and Finance-2 then rebuilt the payout to the real
  vendor account, and a human approved it.
- **Room design fix.** Intake's routing messages carry the employee's request, which is an imperative and scores
  0.998 as outside content. The first ON run therefore falsely tainted IT Access. Routing is now a *delegation*: it
  sets the receiver's task, carries taint, and is not scored. Peer reports are still scored.
- **Python tests.** `cd v2 && offline/.venv/bin/python -m pytest tests/test_room.py -q`: 9/9 pass with a stub
  scorer. They cover health, sticky taint, handoff propagation, fail closed, token auth, the watch level, and
  full room runs: OFF pays the attacker; ON blocks, reroutes and makes no false blocks; a denied approval pays
  nobody.
- **JS tests.** `cd v2/js && npm test`: 10/10 pass against a stub service. They cover the SDK guard, fail closed,
  the gate waiting for an in-flight score, the OpenCode hooks in both shapes, and OpenClaw block and approval.

## Steering: status
- **Round 1** (`v2/offline/out/qwen0.5b/steering_round1.json`): directions fit on 120 + 120 train rows, setting
  picked on 24 val rows, tested on 50 held-out injected and 50 clean rows, greedy decoding.

  | | Steering off | Steering on (layers 6–18, α=3, prompt + generated tokens) |
  |---|---|---|
  | Hijacked (attacker address or secret) | 6/50 | **0/50** |
  | Usable action on injected rows | 50/50 | 29/50 |
  | Clean task completed (steering forced on) | 45/50 | 23/50 |

  Round 1 stops every hijack, but at this strength it breaks about 40% of the model's actions. Its pick rule
  ignored unusable actions, so the strongest setting won by default. **Not a pitch claim as it stands.**
- **Round 2** (`v2/offline/out/qwen0.5b/steering.json`; pick rule: hijacked + unusable actions on val):

  | Val (24 injected / 12 clean) | Hijacked | Usable actions | Clean completed |
  |---|---|---|---|
  | off | 2 | 23 | 8 |
  | L9, α=1 | 1 | 23 | 7 |
  | L6–15, α=2 (picked) | 1 | 24 | 3 |
  | L6–18, α=3 | 0 | 11 | 6 |
  | L6–18, α=3, prompt only | 0 | 12 | 1 |
  | L6–18, α=5, prompt only | 0 | 0 | 0 |
  | L9–12, α=3, prompt only | 1 | 20 | 6 |

  Test for the picked setting: **hijacked 6/50 → 4/50**, usable actions 50/50 → 50/50, clean completed
  **45/50 → 22/50**.
- **Verdict on Qwen 0.5B.** Clamping along the mean injected-minus-clean direction does not separate "follow the
  injection" from "do the task". Either it stops hijacks by breaking the replies, or it barely changes hijacks
  while hurting clean work. The baseline hijack rate is low too (6/50), so there is little to fix. **Do not pitch
  steering as a measured defence.** Pitch the room's macro steering instead: taint propagation, quarantine,
  attribution-based sanitizing, reroute and human approval, all measured in the room runs. Show activation steering
  as an experimental hook (`/v1/chat/completions`, `local` brain).
- **Next for steering, if there is time:**
  - Llama-3B on Garv's GPU, which follows injections more often.
  - Directions from the generated action tokens instead of the last prompt token.
  - Steering conditional on the probe's per-token projection, rather than a fixed clamp.
- Scope: steering edits the agent's own forward pass. It applies only to open-weight agents Anchor hosts (local
  Qwen through `/v1/chat/completions` or the room's `local` brain). For Gemini agents the room steers at the macro
  level: quarantine, sanitize, reroute, human approval.
- In the room, steering only switches on after a session is already tainted, when its side effects are blocked
  anyway. So a damaged reply costs less there than in this test, where steering is forced on.

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
- **Verified on this Mac (2026-11-21).** The real OpenClaw CLI (`openclaw@2026.11.6`, Node 24.22.3, isolated state
  in `v2/js/.cache/harness/oc-state`) installs and enables `anchor-guard`. `plugins inspect anchor-guard --runtime`
  lists the typed hooks `after_tool_call`, `before_agent_run` and `before_tool_call`, with no diagnostics once
  `allowConversationAccess` is set. Without the flag, OpenClaw warns that `before_agent_run` is blocked, as
  documented.
- **OpenCode check.** `@opencode/plugin@2.3.2` types (`dist/effect/tool.d.ts`) match the 2.3.x adapter: the
  `execute.before` and `execute.after` events carry `tool`, `sessionID` and `id`, and the after event has
  `status` plus `result`. A rejection in `execute.before` fails that one call.
- **Not yet done:** a live agent turn that calls tools in either harness. That needs an LLM provider key (for
  example Gemini) or a tools-capable local model. OpenClaw needs Node ≥ 24.16 (OpenClaw itself
  installs under the local Node 24 in `.cache/harness`).
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
1. ~~Long-context eval~~ (done; above). **Retrain the probe on the H1 dataset** (team decision; it rewrites `probe.json`).
2. ~~Steering rounds 1–2~~ (done: no clean win on 0.5B; see verdict). Optional: 3B steering on Garv's GPU.
3. ~~Python tests~~ (done, 9/9). ~~OFF and ON scripted room runs~~ (done). Try the `gemini` and `local` brains.
5. Live OpenCode and OpenClaw runs against the service, if the CLIs can be installed.
6. Pitch deck numbers: only from `long_context.json`, `steering.json` and `v2/room/runs/*.json`.

"""Sidecar scoring: Anchor reads each untrusted tool output in its own pass of the reader model, in the frame the
probe was trained on (one system line, the session's original user request, one tool turn of at most 400 tokens),
never inside the agent's own context.

Why: reading activations inside the agent's generation pass (Option A, managed Qwen serving OpenCode) puts
OpenCode's system prompt, tool schemas and every earlier file read in front of the probe. The injection becomes a
small part of a long prompt and the probe no longer sees the training distribution (measured in
offline/long_context_eval.py → out/<model>/long_context.json).

Long outputs: overlapping windows of MAX_OUTPUT_TOKENS tokens (stride half a window), scored in batches; the
output's score is the highest window score. The parent Scorer only read the first and last window, so an injection
in the middle of a long file was never read.

Sessions: each agent run keeps its task, its untrusted-read scores and a taint flag. Levels:
  tainted — a read scored at or above the probe threshold (from probe.json): every later side effect is blocked;
  watch   — below threshold but a spike against this session's own baseline: later side effects need approval;
  clean   — neither.
The watch rule is advisory and never lowers the measured threshold.
"""
import threading
import time

import numpy as np

from .scorer import MAX_OUTPUT_TOKENS, Scorer, extract

STRIDE = MAX_OUTPUT_TOKENS // 2
BATCH = 8
SPIKE_Z = 4.0          # watch when the score is this many baseline std-devs above the session mean
SPIKE_MIN_STD = 0.02   # floor on the baseline std so a flat baseline does not make every wiggle a spike
SPIKE_MIN_READS = 2    # baseline needs this many earlier clean reads
SPIKE_FLOOR = 0.10     # and the score itself must be at least this


class SidecarScorer(Scorer):
    """Scorer whose raw_score reads every window of a long output, batched. Fail-closed entry point unchanged."""

    def windows(self, output):
        ids = self.tok(output, add_special_tokens=False)["input_ids"]
        if len(ids) <= MAX_OUTPUT_TOKENS:
            return [output]
        starts = list(range(0, len(ids) - MAX_OUTPUT_TOKENS + 1, STRIDE))
        if starts[-1] != len(ids) - MAX_OUTPUT_TOKENS:
            starts.append(len(ids) - MAX_OUTPUT_TOKENS)
        return [self.tok.decode(ids[s:s + MAX_OUTPUT_TOKENS]) for s in starts]

    def _batch_states(self, task, windows, prior):
        """Probe-layer state at the assistant-start token for each window, one padded batch at a time."""
        import torch

        out = []
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        for i in range(0, len(windows), BATCH):
            seqs = [extract.encode(self.tok, extract.messages(task, w, extract.FETCH_CALL, tuple(prior)))[1]
                    for w in windows[i:i + BATCH]]
            n = max(map(len, seqs))
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), n), dtype=torch.long)
            for j, s in enumerate(seqs):  # right padding: real tokens keep positions 0..len-1
                ids[j, :len(s)] = torch.tensor(s)
                mask[j, :len(s)] = 1
            with torch.no_grad():
                hs = self.model(ids.to(self.device), attention_mask=mask.to(self.device),
                                output_hidden_states=True).hidden_states[self.layer]
            for j, s in enumerate(seqs):
                out.append(hs[j, len(s) - 1].float().cpu().numpy())
        return out

    def raw_score(self, task, output, prior=()):
        if not output.strip():
            return {"drift_score": 0.0, "skipped": True, "truncated": False, "windows": 0}
        if self.position != "start":  # the batched path reads the assistant-start token only
            return super().raw_score(task, output, prior)
        wins = self.windows(output)
        prior = [self.windows(p)[-1] for p in prior][-1:]  # split payloads: the previous page's last window
        base = self._base(task)
        scores = [round(self._prob(s - base), 6) for s in self._batch_states(task, wins, prior)]
        k = int(np.argmax(scores))
        return {"drift_score": scores[k], "skipped": False, "truncated": len(wins) > 1, "windows": len(wins),
                "max_window": k, "window_scores": scores}


class Session:
    """One agent run: the original user request, untrusted-read scores, taint level. Thread-safe."""

    def __init__(self, session_id, task, agent=None):
        self.id, self.task, self.agent = session_id, task, agent
        self.reads, self.events = [], []
        self.level, self.reason, self.tainted_by = "clean", None, None
        self.last_untrusted = None
        self.lock = threading.Lock()

    def baseline(self):
        clean = [r["drift_score"] for r in self.reads if r["level"] == "clean" and r["drift_score"] is not None]
        if len(clean) < SPIKE_MIN_READS:
            return None
        return float(np.mean(clean)), max(float(np.std(clean)), SPIKE_MIN_STD)

    def classify(self, result, threshold):
        """Level for one read result; taint is sticky (a run never becomes cleaner)."""
        s = result.get("drift_score")
        if result.get("fail_closed") or (s is not None and s >= threshold):
            return "tainted", None
        b = self.baseline()
        if b and s is not None and s >= SPIKE_FLOOR:
            z = (s - b[0]) / b[1]
            if z >= SPIKE_Z:
                return "watch", round(z, 2)
            return "clean", round(z, 2)
        return "clean", None

    def record(self, source, result, threshold, raised_by=None):
        with self.lock:
            level, z = self.classify(result, threshold)
            read = {"t": time.time(), "source": source, "drift_score": result.get("drift_score"), "spike_z": z,
                    "level": level, "fail_closed": result.get("fail_closed", False), "reason": result.get("reason")}
            self.reads.append(read)
            rank = {"clean": 0, "watch": 1, "tainted": 2}
            if rank[level] > rank[self.level]:
                self.level, self.tainted_by = level, raised_by or source
                self.reason = result.get("reason") or (f"drift {read['drift_score']} ≥ threshold {threshold:.3f}"
                                                       if level == "tainted" else f"spike z={z} vs session baseline")
            return read

    def taint_from(self, other_id, reason):
        """Macro rule: a message from a tainted agent taints the receiver."""
        with self.lock:
            if self.level != "tainted":
                self.level, self.tainted_by, self.reason = "tainted", other_id, reason

    def decide(self, tool):
        """Side-effect gate: allow / approve / block, from the session level."""
        if self.level == "tainted":
            return {"decision": "block", "reason": self.reason, "tainted_by": self.tainted_by}
        if self.level == "watch":
            return {"decision": "approve", "reason": self.reason, "tainted_by": self.tainted_by}
        return {"decision": "allow", "reason": None, "tainted_by": None}

    def snapshot(self):
        with self.lock:
            return {"id": self.id, "agent": self.agent, "task": self.task, "level": self.level, "reason": self.reason,
                    "tainted_by": self.tainted_by, "reads": list(self.reads)}

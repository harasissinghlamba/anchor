"""Activation steering: when Anchor taints a run, push the agent model's residual stream back off the injection
direction while it reads the poisoned output and chooses its next action.

Direction per layer L (fit on train rows, generation frame): d_L = mean(h | injected) − mean(h | clean) at the
last prompt token, unit-normalised; mu_L = mean clean projection on d_L. Intervention on every position from the
start of the tool output onward (prefill) and on every generated token:

    h ← h − alpha · max(0, ⟨h, d_L⟩ − mu_L) · d_L

i.e. clamp the injection component down to the clean mean (alpha = 1) or past it (alpha > 1). Only the excess
along d_L is removed, so a state already at the clean level is left alone. This needs the agent's own forward pass:
it applies to open-weight agents Anchor hosts, not to closed APIs. Measured in offline/steer_eval.py.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot find decoder layers on this model")


class Steerer:
    def __init__(self, model, directions, alpha=1.0):
        """directions: {layer: (unit vector np.ndarray, mu float)}; hidden_states[L] = output of decoder layer L−1."""
        import torch

        self.model, self.alpha = model, alpha
        self.on_generated = True  # False: steer only the prefill (how the poisoned output is read)
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        self.dirs = {L: (torch.tensor(d, device=dev, dtype=dt), float(mu)) for L, (d, mu) in directions.items()}

    # --- fitting -----------------------------------------------------------------------------------------------------
    @staticmethod
    def fit(states_inj, states_clean):
        """{layer: [vectors]} for each class → {layer: (unit d, mu_clean)}."""
        out = {}
        for L in states_inj:
            a, b = np.stack(states_inj[L]), np.stack(states_clean[L])
            d = a.mean(0) - b.mean(0)
            d = d / (np.linalg.norm(d) + 1e-8)
            out[L] = (d.astype(np.float32), float((b @ d).mean()))
        return out

    def save(self, path, meta):
        Path(path).write_text(json.dumps({**meta, "alpha": self.alpha, "on_generated": self.on_generated, "layers": {
            str(L): {"d": d.float().cpu().numpy().round(6).tolist(), "mu": mu} for L, (d, mu) in self.dirs.items()}}))

    @classmethod
    def load(cls, model, path, alpha=None):
        j = json.loads(Path(path).read_text())
        dirs = {int(L): (np.array(v["d"], dtype=np.float32), v["mu"]) for L, v in j["layers"].items()}
        st = cls(model, dirs, j["alpha"] if alpha is None else alpha)
        st.on_generated = j.get("on_generated", True)
        return st, j

    # --- intervention ---------------------------------------------------------------------------------------------
    @contextmanager
    def active(self, start_pos=0, log=None, on_generated=True):
        """Hooks on; positions ≥ start_pos in the prefill (and every generated token unless on_generated=False:
        steer only how the poisoned output is read, not how the reply is written) are steered.
        log (list) receives per-layer removed magnitude for the dashboard."""
        layers = decoder_layers(self.model)
        handles = []

        def make(L, d, mu):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] == 1 and not on_generated:
                    return out
                seg = h[:, start_pos:, :] if h.shape[1] > 1 else h
                proj = seg @ d
                excess = (proj - mu).clamp(min=0)
                seg.sub_(self.alpha * excess.unsqueeze(-1) * d)
                if log is not None:
                    log.append({"layer": L, "removed": float(excess.mean())})
                return out
            return hook

        for L, (d, mu) in self.dirs.items():
            handles.append(layers[L - 1].register_forward_hook(make(L, d, mu)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()


def output_start(tok, text, offsets, output):
    """Token index where the tool output begins in the rendered prompt (0 if it cannot be found)."""
    for form in (output, json.dumps(output, ensure_ascii=False)[1:-1]):
        head = form[:40]
        at = text.rfind(head) if head else -1
        if at >= 0:
            return min(i for i, (s, e) in enumerate(offsets) if e > at)
    return 0

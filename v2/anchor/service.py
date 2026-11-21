"""Anchor service: the HTTP API behind @anchor-ai/sdk, the OpenCode / OpenClaw plugins and the Control Room.

  GET  /v1/health                      model, layer, threshold (from probe.json), device, steering available
  POST /v1/sessions   {task, session_id?, agent?}            → {session_id}
  POST /v1/score      {session_id, source, content, attribution?}
                      sidecar score in Anchor's own frame → read level, session level, spike z, fail_closed
  POST /v1/decide     {session_id, tool}                      → {decision: allow|approve|block, reason, tainted_by}
  POST /v1/handoff    {from, to, content}   macro rule: a tainted sender taints the receiver; content is scored
  GET  /v1/sessions/{id}, GET /v1/room     snapshots;  GET /v1/events  server-sent events for dashboards
  POST /v1/chat/completions   OpenAI-compatible generation by the reader model; when the session named in the
                      `x-anchor-session` header is tainted and steering directions exist, steering is on.

Run: cd v2 && ANCHOR_DEVICE=mps python -m anchor.service   (env ANCHOR_MODEL, ANCHOR_PORT, ANCHOR_TOKEN)
Fail closed everywhere: scorer errors taint the session; unknown sessions are created with a generic task.
"""
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2 / "offline") not in sys.path:
    sys.path.insert(0, str(V2 / "offline"))

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .sidecar import Session  # noqa: E402

MAX_CONTENT = 400_000  # characters; larger outputs are cut (and say so) rather than refused


class SessionIn(BaseModel):
    task: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=200)
    agent: str | None = Field(default=None, max_length=100)


class ScoreIn(BaseModel):
    session_id: str = Field(max_length=200)
    source: str = Field(default="tool", max_length=200)
    content: str
    attribution: bool = False


class DecideIn(BaseModel):
    session_id: str = Field(max_length=200)
    tool: str = Field(max_length=200)


class HandoffIn(BaseModel):
    from_: str = Field(alias="from", max_length=200)
    to: str = Field(max_length=200)
    content: str = ""


class Hub:
    """Sessions + event fan-out. Events are dicts; subscribers get them as SSE."""

    def __init__(self, scorer, steerer=None):
        self.scorer, self.steerer = scorer, steerer
        self.sessions: dict[str, Session] = {}
        self.events: list[dict] = []
        self.subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self.lock = threading.Lock()

    def emit(self, kind, **data):
        ev = {"t": round(time.time(), 3), "kind": kind, **data}
        with self.lock:
            self.events.append(ev)
            del self.events[:-2000]
            subs = list(self.subs)
        for loop, q in subs:
            loop.call_soon_threadsafe(q.put_nowait, ev)
        return ev

    def session(self, sid, task=None, agent=None):
        with self.lock:
            s = self.sessions.get(sid)
            if s is None:
                s = self.sessions[sid] = Session(sid, task or "Complete the user's request.", agent)
                created = True
            else:
                created = False
        if created:
            self.emit("session", session=sid, agent=agent, task=s.task)
        return s

    def score(self, s, source, content, attribution=False, raised_by=None):
        truncated = len(content) > MAX_CONTENT
        # Each read is scored on its own (the single-page training frame); split payloads across reads are a
        # known gap of this path (the offline split rows carry one prior page).
        r = self.scorer.score(s.task, content[:MAX_CONTENT], attribution="projection" if attribution else None)
        read = s.record(source, r, self.scorer.threshold, raised_by=raised_by)
        out = {**r, "level": read["level"], "session_level": s.level, "spike_z": read["spike_z"],
               "content_truncated": truncated}
        out.pop("window_scores", None)
        self.emit("score", session=s.id, agent=s.agent, source=source, drift_score=r.get("drift_score"),
                  level=read["level"], session_level=s.level, spike_z=read["spike_z"],
                  fail_closed=r.get("fail_closed", False), latency_ms=r.get("latency_ms"),
                  preview=content[:160])
        return out


    def handoff(self, src, dst, content):
        """Macro rule: a tainted sender taints the receiver; the message itself is scored as an untrusted read."""
        if src.level == "tainted":
            dst.taint_from(src.id, f"message from tainted agent {src.agent or src.id}")
            self.emit("propagate", session=dst.id, agent=dst.agent, source=src.id, source_agent=src.agent)
        out = self.score(dst, f"message:{src.agent or src.id}", content, raised_by=src.id) if content.strip() \
            else {"level": dst.level}
        out["session_level"] = dst.level
        return out


def create_app(scorer=None, steerer=None, model_info=None, static_dir=None):
    """scorer: object with .score(task, output, prior, attribution) and .threshold (SidecarScorer in production;
    tests pass their own). Loaded from probe.json when None."""
    if scorer is None:
        from common import MODELS, Run
        from .sidecar import SidecarScorer
        name = os.environ.get("ANCHOR_MODEL", "qwen0.5b")
        run = Run(name, MODELS[name]["default_precision"])
        scorer = SidecarScorer(run)
        dirs = run.out / "steer_dirs.json"
        if steerer is None and dirs.exists():
            from .steer import Steerer
            steerer, _ = Steerer.load(scorer.model, dirs)
        model_info = {"model": run.hf_id, "device": str(scorer.device)}
    hub = Hub(scorer, steerer)
    token = os.environ.get("ANCHOR_TOKEN")
    app = FastAPI(title="Anchor")
    app.state.hub = hub

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token and request.url.path.startswith("/v1/") and request.url.path != "/v1/health":
            if request.headers.get("authorization") != f"Bearer {token}":
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/v1/health")
    def health():
        info = model_info or {}
        return {"ok": True, "model": info.get("model", "unknown"), "device": info.get("device", "unknown"),
                "layer": getattr(scorer, "layer", None), "threshold": scorer.threshold,
                "steering": steerer is not None, "sessions": len(hub.sessions)}

    @app.post("/v1/sessions")
    def new_session(body: SessionIn):
        sid = body.session_id or uuid.uuid4().hex[:12]
        s = hub.session(sid, body.task, body.agent)
        return {"session_id": s.id, "task": s.task}

    @app.post("/v1/score")
    def score(body: ScoreIn):
        s = hub.session(body.session_id)
        return hub.score(s, body.source, body.content, body.attribution)

    @app.post("/v1/decide")
    def decide(body: DecideIn):
        s = hub.session(body.session_id)
        d = s.decide(body.tool)
        hub.emit("gate", session=s.id, agent=s.agent, tool=body.tool, **d)
        return d

    @app.post("/v1/handoff")
    def handoff(body: HandoffIn):
        return hub.handoff(hub.session(body.from_), hub.session(body.to), body.content)

    @app.get("/v1/sessions/{sid}")
    def get_session(sid: str):
        s = hub.sessions.get(sid)
        if s is None:
            raise HTTPException(404, "no such session")
        return s.snapshot()

    @app.get("/v1/room")
    def room():
        return {"sessions": [s.snapshot() for s in hub.sessions.values()], "threshold": scorer.threshold,
                "events": hub.events[-200:]}

    @app.get("/v1/events")
    async def events():
        loop, q = asyncio.get_running_loop(), asyncio.Queue()
        with hub.lock:
            hub.subs.append((loop, q))
            backlog = hub.events[-200:]

        async def stream():
            try:
                for ev in backlog:
                    yield f"data: {json.dumps(ev)}\n\n"
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                with hub.lock:
                    hub.subs.remove((loop, q))
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(request: Request, x_anchor_session: str | None = Header(default=None)):
        body = await request.json()
        if not hasattr(scorer, "model"):
            raise HTTPException(503, "generation needs the reader model")
        s = hub.sessions.get(x_anchor_session) if x_anchor_session else None
        steer = bool(steerer and s and s.level == "tainted")
        text = await asyncio.to_thread(generate, scorer, steerer if steer else None, body.get("messages", []),
                                       int(body.get("max_tokens") or 256))
        hub.emit("generate", session=x_anchor_session, steered=steer)
        return {"id": "anchor-" + uuid.uuid4().hex[:8], "object": "chat.completion", "created": int(time.time()),
                "model": (model_info or {}).get("model", "anchor"), "anchor": {"steered": steer},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}

    try:
        from room.runner import mount
        mount(app, hub)
    except ImportError:
        pass

    static = Path(static_dir) if static_dir else V2 / "room" / "public"
    if static.exists():
        @app.get("/")
        def index():
            return FileResponse(static / "index.html")

        from fastapi.staticfiles import StaticFiles
        app.mount("/room", StaticFiles(directory=static), name="room")
    return app


def generate(scorer, steerer, messages, max_new=256):
    """Greedy generation by the reader model; steering hooks on from the first tool message when steerer is set."""
    import torch
    from common import render
    from .steer import output_start

    tok, model, device = scorer.tok, scorer.model, scorer.device
    msgs = [{"role": m.get("role", "user"), "content": m.get("content") if isinstance(m.get("content"), str)
             else json.dumps(m.get("content"))} for m in messages]
    text = render(tok, msgs)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    first_tool = next((m["content"] for m in msgs if m["role"] == "tool"), None)
    start = output_start(tok, text, enc["offset_mapping"], first_tool) if first_tool else 0
    x = torch.tensor([enc["input_ids"]], device=device)
    with scorer.lock, torch.no_grad():
        if steerer is not None:
            with steerer.active(start_pos=start, on_generated=steerer.on_generated):
                g = model.generate(x, max_new_tokens=max_new, do_sample=False)
        else:
            g = model.generate(x, max_new_tokens=max_new, do_sample=False)
    return tok.decode(g[0, x.shape[1]:], skip_special_tokens=True)


def main():
    import uvicorn
    port = int(os.environ.get("ANCHOR_PORT", "8765"))
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

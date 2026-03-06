"""OpenAI-compatible API server for ANEMLL models."""

import asyncio
import json
import queue
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import ANEEngine


app = FastAPI(title="anemll")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set by serve_model()
_engine: ANEEngine | None = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1000
    stream: bool = False


class _TokenGenerator:
    """Runs token generation in a background thread for async streaming."""

    def __init__(self, engine: ANEEngine, messages: list[dict], temperature: float, max_tokens: int):
        self.engine = engine
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            for token_text in self.engine.generate(self.messages, self.temperature, self.max_tokens):
                if self.stop_event.is_set():
                    break
                self.token_queue.put(token_text)
        except Exception as e:
            self.token_queue.put({"error": str(e)})
        finally:
            self.token_queue.put(None)  # sentinel

    def stop(self):
        self.stop_event.set()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            _stream(request, messages),
            media_type="text/event-stream",
        )

    # Non-streaming
    content = ""
    for token in _engine.generate(messages, request.temperature, request.max_tokens):
        content += token

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


async def _stream(request: ChatRequest, messages: list[dict]):
    gen = _TokenGenerator(_engine, messages, request.temperature, request.max_tokens)
    gen.start()

    cid = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model = request.model

    def chunk(delta: dict, finish: str | None = None):
        return f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"

    yield chunk({"role": "assistant"})

    try:
        empty_count = 0
        while True:
            try:
                token = gen.token_queue.get_nowait()
            except queue.Empty:
                empty_count += 1
                await asyncio.sleep(0.05 if empty_count > 5 else 0.01)
                continue

            empty_count = 0

            if token is None:
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
                break

            if isinstance(token, dict) and "error" in token:
                yield chunk({"content": f"Error: {token['error']}"}, "error")
                yield "data: [DONE]\n\n"
                break

            yield chunk({"content": token})
    finally:
        gen.stop()


@app.get("/v1/models")
async def list_models():
    name = _engine.model_name if _engine else "anemll-model"
    return {
        "object": "list",
        "data": [{
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "anemll",
            "permission": [],
            "root": name,
            "parent": None,
        }],
    }


def serve_model(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """Load a model and start the API server."""
    import uvicorn

    global _engine
    print(f"Loading model from {model_path}...", flush=True)
    _engine = ANEEngine(model_path)
    print(f"Server starting at http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port)

# mini_vllm/server/api_server.py
import asyncio
import json
import queue
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.utils.config import EngineConfig, SamplingParams
from mini_vllm.server.openai_types import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatCompletionChunk, ChatCompletionChunkChoice, DeltaMessage, ChatMessage,
    CompletionRequest, CompletionChunk, CompletionChunkChoice,
    Usage, ModelCard, ModelList,
)
from mini_vllm.utils.logger import logger

app = FastAPI(title="Mini-VLLM API Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# model_name -> LLMEngine
engines: dict[str, LLMEngine] = {}
default_model: str = ""


def _resolve_engine(model: str) -> LLMEngine:
    """Resolve model name to engine. Empty/unknown -> default model."""
    if not model:
        return engines[default_model]
    if model in engines:
        return engines[model]
    # Try case-insensitive partial match (e.g. user sends "qwen3" matches "Qwen3-0.6B")
    model_lower = model.lower()
    for name, eng in engines.items():
        if model_lower in name.lower() or name.lower() in model_lower:
            return eng
    raise HTTPException(status_code=404, detail=f"Model '{model}' not found. Available: {list(engines.keys())}")


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[
        ModelCard(id=name, owned_by="mini-vllm")
        for name in engines
    ])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    engine = _resolve_engine(request.model)

    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt = engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop or [],
    )

    model_name = request.model or default_model

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(engine, model_name, prompt, sampling_params),
            media_type="text/event-stream",
        )

    # Non-streaming
    results = engine.generate([prompt], sampling_params)
    result = results[0]

    return ChatCompletionResponse(
        model=model_name,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=result.output_text),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=len(result.prompt_token_ids),
            completion_tokens=len(result.output_token_ids),
            total_tokens=len(result.prompt_token_ids) + len(result.output_token_ids),
        ),
    )


async def _stream_chat_completion(engine: LLMEngine, model: str, prompt: str, sampling_params: SamplingParams):
    """Streaming output for /v1/chat/completions"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    first_chunk = True
    q: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def _produce():
        with engine._lock:
            try:
                for item in engine.generate_stream([prompt], sampling_params):
                    q.put(item)
            except Exception as e:
                q.put(e)
            finally:
                q.put(_SENTINEL)

    threading.Thread(target=_produce, daemon=True).start()
    loop = asyncio.get_event_loop()

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            _, token_text, finished = item

            if first_chunk:
                role_chunk = ChatCompletionChunk(
                    id=chunk_id, model=model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )],
                )
                yield f"data: {role_chunk.model_dump_json(ensure_ascii=False)}\n\n"
                first_chunk = False

            chunk = ChatCompletionChunk(
                id=chunk_id, model=model,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(content=token_text),
                    finish_reason=None,
                )],
            )
            yield f"data: {chunk.model_dump_json(ensure_ascii=False)}\n\n"

        # End marker
        chunk = ChatCompletionChunk(
            id=chunk_id, model=model,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop",
            )],
        )
        yield f"data: {chunk.model_dump_json(ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


async def _stream_completion(engine: LLMEngine, model: str, prompt: str, sampling_params: SamplingParams):
    """Streaming output for /v1/completions"""
    chunk_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    q: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def _produce():
        with engine._lock:
            try:
                for item in engine.generate_stream([prompt], sampling_params):
                    q.put(item)
            except Exception as e:
                q.put(e)
            finally:
                q.put(_SENTINEL)

    threading.Thread(target=_produce, daemon=True).start()
    loop = asyncio.get_event_loop()

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            _, token_text, finished = item

            chunk = CompletionChunk(
                id=chunk_id, model=model,
                choices=[CompletionChunkChoice(index=0, text=token_text)],
            )
            yield f"data: {chunk.model_dump_json(ensure_ascii=False)}\n\n"

        # End marker
        chunk = CompletionChunk(
            id=chunk_id, model=model,
            choices=[CompletionChunkChoice(index=0, text="", finish_reason="stop")],
        )
        yield f"data: {chunk.model_dump_json(ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    engine = _resolve_engine(request.model)
    model_name = request.model or default_model

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    if request.stream:
        return StreamingResponse(
            _stream_completion(engine, model_name, request.prompt, sampling_params),
            media_type="text/event-stream",
        )

    results = engine.generate([request.prompt], sampling_params)
    result = results[0]

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "model": model_name,
        "choices": [{"text": result.output_text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(result.output_token_ids),
            "total_tokens": len(result.prompt_token_ids) + len(result.output_token_ids),
        },
    }


def main():
    global default_model

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="单个模型路径，或逗号分隔的多个模型路径")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备: auto (自动检测), cuda (GPU), cpu")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()

    # Parse model paths: support comma-separated or single
    model_paths = [p.strip() for p in args.model_path.split(",") if p.strip()]

    for path in model_paths:
        if not Path(path).exists():
            logger.error(f"Model path not found: {path}")
            continue

        config = EngineConfig(
            model_path=path,
            device=args.device,
            dtype=args.dtype,
            max_num_seqs=args.max_num_seqs,
        )
        engine_inst = LLMEngine(config)

        # Use directory name as model name (matches client-side usage)
        model_name = Path(path).name
        # Deduplicate if name collision
        if model_name in engines:
            model_name = f"{model_name}_{len(engines)}"

        engines[model_name] = engine_inst

        if not default_model:
            default_model = model_name

    if not engines:
        logger.error("No models loaded! Check --model-path")
        return

    logger.info(f"Models: {list(engines.keys())}")
    logger.info(f"Server: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

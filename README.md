# llm-sdk

A small Python convenience layer for local and OpenAI-compatible LLM endpoints.

The SDK gives you one interface for text, reasoning, tools and media inputs. It
uses the official OpenAI client for HTTP, retries and connection management.
It does not run tools or manage agents, conversations, routing or billing.

Requires Python 3.10+ and OpenAI Python 2.x. Install:

```sh
pip install llm-sdk-py
# Optional image resizing and transcoding:
pip install "llm-sdk-py[pillow]"
```

You can also copy `llm_sdk.py` into your project and install its dependencies.

## Local inference

Load a model in your local server and use its exact model ID:

```python
from llm_sdk import LLM

with LLM(model="your-loaded-model", base_url="http://localhost:1234/v1") as llm:
    response = llm.response(input="Write a tiny haiku about fast code.")
    print(response["answer"])
    print(response["usage"])
```

Defaults are `base_url="http://localhost:1234/v1"` and `api_key="lm-studio"`.
Pathless URLs get `/v1`; use `normalize_base_url=False` to preserve a custom
base path. URLs must use HTTP(S) without embedded credentials or fragments.

All inference methods accept either `input="..."` or `messages=[...]`, plus an
optional `system="..."`. A supplied `system` replaces an existing leading
system message. The client keeps no conversation history: send previous
messages yourself. Use a context manager to close the client.

## Streaming and async

```python
from contextlib import closing

with LLM(model="your-loaded-model") as llm:
    with closing(llm.stream_response(input="Explain adapters briefly.", final=True)) as stream:
        for event in stream:
            if event["type"] == "answer":
                print(event["content"], end="", flush=True)
            elif event["type"] == "final":
                print(event["content"]["usage"])
```

Events are dictionaries with `type` and `content`. Types are `answer`,
`reasoning`, `refusal`, `tool_call_part`, `tool_call`, `verbose`, `final` and
`done`. `final=True` adds the aggregated result before `done`.
Structured answers are emitted once, after parsing, rather than as JSON fragments.

Use `async_response` / `async_stream_response` for async calls:

```python
import asyncio
from contextlib import aclosing
from llm_sdk import LLM

async def main():
    async with LLM(model="your-loaded-model") as llm:
        response = await llm.async_response(input="Give me a short project name.")
        print(response["answer"])
        async with aclosing(llm.async_stream_response(input="Explain that name.")) as stream:
            async for event in stream:
                if event["type"] == "answer":
                    print(event["content"], end="", flush=True)

asyncio.run(main())
```

Requests start when the generator is first consumed. Close a stream on early
exit with `close()` or `await aclose()`; `closing` / `aclosing` handles this
on `break`, exceptions and normal completion. A plain `break` is insufficient.
Use separate instances for threads running separate async event loops.

`response()` and `async_response()` aggregate an HTTP stream internally, so
these methods also require an endpoint that supports streaming.

## Structured output

For locally validated output, use a **Pydantic v2** model:

```python
from pydantic import BaseModel, Field

class Review(BaseModel):
    sentiment: str
    score: int = Field(ge=1, le=5)

with LLM(model="your-loaded-model") as llm:
    response = llm.response(input="Review: fast, small, surprisingly nice.", output_format=Review)
    print(response["answer"])  # dict, not a Review instance
```

Pydantic aliases, constraints and nested models use the OpenAI SDK's schema
conversion. The returned JSON is validated locally with
`model_validate_json(..., strict=True)` before any answer is emitted. The
answer stays JSON-compatible data, including aliases; validation does not
replace it with a model instance or fill omitted defaults. JSON-native
representations such as ISO date strings remain supported by Pydantic.

Plain annotated classes, dataclasses, TypedDicts and OpenAI `response_format`
dictionaries are also accepted. For these, the SDK generates/passes the schema
and parses JSON, **without local schema validation**. The server must enforce
the schema. Recursive plain classes and recursive models embedded in plain
classes or callable parameters are unsupported; use a Pydantic output model
at the root for recursive output schemas.

Malformed JSON, refusals and Pydantic validation failures raise
`StructuredOutputError`; `.raw` contains the original answer and `.stop_reason`
explains termination when available. Tools-only turns may have `answer=None`.
Outer Markdown fences are accepted; prose around JSON is rejected.
`schema_strict=False` relaxes the server schema; JSON parsing and Pydantic
validation remain strict.

## Tools

Pass Python callables or OpenAI function-tool definitions through `tools=[...]`.
The SDK generates callable schemas and returns calls; **your application decides
which functions to execute**. Invalid arguments can be returned as `{"_raw": ...}`;
validate arguments before execution. Completed calls have `id`, `name`,
`arguments` and an optional `callable` reference. Only execute tools you allow.

Use `assistant_message(response)` and `tool_result(call, value)` to append a
completed tool turn to your messages. `user_message` and `system_message` are
also available. Responses-mode reasoning items are carried by
`assistant_message` for replay. See [the full tool dialog](examples/tools.py).

## Images, audio, video and files

Use these parts in a user message's `content` list:

- Image: `{"type": "image", "image_path": "photo.png"}`; also `image_url`, `image_base64`, `image_pil`.
- Audio: `{"type": "audio", "audio_path": "clip.wav"}`; also `audio_url`, `audio_base64`.
- Video: `{"type": "video", "video_path": "clip.mp4"}`; also `video_url`, `video_base64`.
- File: `{"type": "file", "file_path": "doc.pdf"}`; also `file_url`, `file_base64`, `file_id`.

Example:

```python
with LLM(model="your-vision-model") as llm:
    response = llm.response(messages=[{"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image", "image_path": "photo.png"},
    ]}])
```

Paths may be strings or `pathlib.Path`. Pillow is needed for PIL input,
transcoding and resizing. Images are capped at 8192 px on the longest side
(configurable with `max_image_side`, `None` disables the side cap). Resizing
preserves aspect ratio; an image that still exceeds the byte budget raises
`ImageProcessingError`. Pillow's global limits are left unchanged.

Inline limits: images 15 MiB sent / 100 MP decoded, audio 25 MiB,
video 100 MiB and files 50 MiB. Limits apply per item, not to the entire request.
Remote URLs pass to the provider; the SDK does not download them. Native file
parts are validated too. Input format support depends on the endpoint and model.

## OpenAI and Responses mode

```python
import os

with LLM(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.openai.com/v1",
    use_responses_api=True,
) as llm:
    response = llm.response(input="Give me a crisp project name.")
    print(response["answer"])
```

Responses mode translates messages, function tools, schemas and supported
media into Responses API input. This adapter rejects video and remote audio
URLs; it supports base64 audio and file URLs/IDs. It is a convenience subset,
not a replacement for all native OpenAI API features. Use the official client
when you need native resources or event types.

`store=False` includes encrypted reasoning for replay by default. An explicit
`extra_body={"include": [...]}` overrides this; keep
`"reasoning.encrypted_content"` if your next turn needs those items.

## Options, usage and errors

- `max_retries=3` by default, configurable globally and per call. Retries use the OpenAI client; consumed stream events are never replayed automatically.
- `timeout=300` seconds globally; `None` disables the timeout.
- `reasoning_effort="high"` or `reasoning_budget=2000`, mutually exclusive. Reasoning visibility defaults to on; `include_reasoning=False` hides its text, not the computation. `reasoning_unterminated` still reports an incomplete tagged block.
- Generation options: `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `user`, `tool_choice`, `store`. Explicit sampling values pass through; choose values your model supports. Only on `api.openai.com`, Chat `max_tokens` maps to `max_completion_tokens`. Other hosts retain `max_tokens` regardless of model name. Responses maps it to `max_output_tokens` and warns when ignoring `seed`/`stop`.
- `extra_body` deep-merges provider options, last, and may override normal fields. `model`, `messages`, `input` and `stream` are reserved. Use it for endpoint-specific token-limit or reasoning fields when necessary.
- Every final result has `usage` with `prompt_tokens`, `completion_tokens` and `total_tokens`. Missing counts are `None`; totals are derived only if both counts are known.
- `verbose=True` adds stream statistics. Without server usage, `tokens` falls back to **chunk count**, flagged by `tokens_is_estimated=True`. Throughput includes consumer pauses; a one-chunk response uses full elapsed time. These figures are diagnostic, not billing data.
- API failures raise `ModelRequestError` with available status, request ID and body; configuration, schema and media errors have separate exception types.
- Logging is quiet by default. `debug=True` changes the SDK logger process-wide. `configure_debug_logging()` / `configure_quiet_logging()` are explicit logging helpers.
- `LLMConfig` / `LLM.from_config` support config-object construction. Config values are copied at construction.
- `list_models` / `async_list_models` exist on clients and as standalone functions. Use `raise_on_error=True` when listing must fail rather than log and return the fallback.

## Development and maintenance

```sh
python -m pip install -e ".[dev,pillow]"
python -m pytest -q
python -m ruff check .
python -m build
```

Executable examples: [local](examples/local.py), [streaming](examples/streaming.py),
[structured output](examples/structured.py), [tool dialog](examples/tools.py).
Set `LLM_MODEL`, optionally `LLM_BASE_URL` and `LLM_API_KEY`, then run e.g.
`python examples/local.py`. Never commit API keys.

See [compatibility and manual checks](COMPATIBILITY.md) and [release notes](CHANGELOG.md).
The intended scope is stable inference helpers. Maintenance focuses on bugs,
dependencies and demonstrated endpoint compatibility issues. New provider
frameworks and agent orchestration are outside that scope.

## License

MIT

# Compatibility

The supported contract is OpenAI-compatible streaming inference, not every
feature exposed by the official SDK or every vendor's interpretation of it.

## Automated coverage

Tests exercise the real OpenAI client against an in-process HTTPX mock transport,
including JSON request serialization and SSE parsing. Other unit tests cover
media normalization, schema conversion, reasoning and tool-event handling.
These are protocol tests, not evidence of live provider compatibility.

| Area | Contract |
| --- | --- |
| Chat Completions | Sync/async, aggregated/streaming, tools, usage, structured output |
| Responses adapter | Sync/async, aggregated/streaming, translated tool history, reasoning replay |
| Pydantic v2 output | Native schema conversion plus strict local JSON validation |
| Plain classes / raw schemas | Schema generation/passthrough plus JSON parsing; server validates schema |
| Media | Local normalization and rejection tests; server/model format support varies |
| Dependencies | OpenAI `>=2.0.0,<3`; optional Pillow `>=10,<13` |

CI runs Python 3.10–3.13 on Linux, Python 3.11 on Windows and an additional
OpenAI 2.0.0 job. Passing CI does not certify untested provider versions.

## Manual checks before releasing endpoint compatibility changes

Install the checkout with `python -m pip install -e ".[dev,pillow]"`. Start
your server and load an appropriate model. In PowerShell:

```powershell
$env:LLM_MODEL = 'your-loaded-model-id'
$env:LLM_BASE_URL = 'http://localhost:1234/v1'
$env:LLM_API_KEY = 'lm-studio'
python examples/local.py
python examples/streaming.py
python examples/structured.py
python examples/tools.py
```

Use the exact model ID reported by your server. For LM Studio, vLLM, llama.cpp
or another server, record its version, model ID, endpoint and which examples
passed. Only claim support for combinations actually checked. Structured-output
and tool examples require a model/server that supports those capabilities.
An endpoint must support streaming even when calling `response()`.

For OpenAI Responses, set `OPENAI_MODEL` and `OPENAI_API_KEY` and run the
Responses example in the README. Verify text, a function-tool round trip and
structured output with the model you use. Do not assume that a successful
Chat Completions call verifies Responses mode.

No live provider/version combination is certified by this document. Record
actual live checks in release notes when performed; do not add private prompts
or credentials.

## Deliberate boundaries

- No native Anthropic/Gemini adapter, agent runner, routing, pricing database or
  independent retry framework. Use another client or gateway for those needs.
- No arbitrary JSON repair or extraction from prose. Invalid output raises.
- No local schema validation for plain classes or raw JSON schemas. Use
  Pydantic v2 output models when local validation is required.
- Reasoning formats and media limits vary by endpoint. The Responses adapter
  currently rejects video and remote audio URLs. It preserves supported
  reasoning items via `assistant_message` for multi-turn replay.
- Timeout/retries are owned by the OpenAI client. A stream that already emitted
  output is not transparently restarted. Applications decide how to recover.
- Responses, Chat input shapes and stream event dictionaries remain the public
  abstraction; native OpenAI resources should be accessed with the official SDK.

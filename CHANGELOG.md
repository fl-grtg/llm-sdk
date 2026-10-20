# Changelog

## 2.1.0 — unreleased

Stabilize the existing inference API without introducing another framework or
additional runtime dependencies.

### Changed

- Pydantic v2 output models use the OpenAI client's schema conversion, preserving
  aliases, constraints and nested definitions. All four inference methods validate
  the returned JSON locally in strict mode. Answers remain JSON-compatible data.
- Explicit `temperature`, `top_p` and Chat `stop` values are no longer silently
  removed based on model-name guesses. Omit options unsupported by your model.
- Only Chat requests to `api.openai.com` map `max_tokens` to
  `max_completion_tokens`. Other hosts retain `max_tokens`, including locally
  hosted models with names beginning `gpt-5` or `o3`. Use `extra_body` when an
  endpoint needs a different field.
- Native file parts now share validation with convenience inputs and normalize
  inline data into MIME-qualified base64 URLs. Ambiguous file sources are rejected.
- Incomplete tagged reasoning is reported by `reasoning_unterminated` even when
  reasoning text is hidden.

### Added

- `usage` on every aggregated/final result, independent of `verbose`. Missing
  token counts stay `None`; chunk estimates never become total token counts.
- HTTP contract tests for sync/async Chat and Responses requests and SSE streams.
- Opt-in live endpoint contracts covering tools, schemas, images, async
  concurrency, early closure, cancellation, timeouts, authentication and truncation.
- Runnable local, streaming, structured-output and tool-dialog examples.
- CI coverage for the minimum supported OpenAI dependency and Windows.

### Fixed

- Reject open Pydantic object schemas locally in strict mode, including nested
  definitions; `schema_strict=False` remains available for non-strict endpoints.
- Remove obsolete discriminator mappings when inlining embedded schema references.
- Keep different idless tools separate when a server reuses the same index.
- Use absolute README links so documentation and examples also open from PyPI.
- Preserve Pillow's process-wide pixel limit.
- Reject images still exceeding the send budget after resizing.
- Reject Responses tool history with missing call IDs or function names locally.
- Keep unnamed-ID Responses tool items separate by output index and warn on
  unresolvable function-call completions.
- Separate different indexless tool names even when earlier arguments are incomplete.
- Close the nested async Responses stream when the public generator is closed.
- Reject unfinished empty/reasoning-only streams without an answer, completed
  tool call, refusal or terminal status, instead of returning an empty result.
- Accept `output_text` parts in Chat response content lists.
- Respect explicit Responses `include` overrides when `store=False`.
- Remove repeated schema preparation in aggregated calls; correct package URLs
  and document stream cleanup, schema guarantees and adapter limitations.

## 2.0.1

Quieter logging, token-statistics corrections and empty system-message handling.

## 2.0.0

Multimodal inputs, reasoning included by default and stricter core validation.

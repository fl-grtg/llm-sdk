# Live API verification — 2026-10-20

## Scope

Tested the local 2.1.0 source using Python 3.11 and OpenAI Python 2.15.0:

- Google OpenAI-compatible Chat Completions, requested model `gemma-4-31b-it`.
- A locally configured authenticated Responses proxy, requested model
  `gpt-6-luna`, reasoning effort `medium`, `store=False`.

Credentials were loaded into process memory from the environment and the
authorized neighboring project's environment file. No keys, private prompts,
environment file contents or private proxy configuration are included here.
Requests used arithmetic, colors, generated red images and a local addition tool.

## Successful live checks

Both endpoints completed sync/async text calls, sync streaming, locally validated
Pydantic output (aliases and numeric constraints), sync/async function-tool
round trips, image input and three concurrent isolated async requests.
Both passed their explicit async streaming contract. For Google, the final
successful async-stream check used `reasoning_effort="minimal"`, 4096 output
tokens and three retries; earlier automatic-reasoning runs were unreliable.

Both passed early stream closure followed by client reuse, invalid-credential
error typing, deliberately short timeouts and task cancellation followed by
reuse. Google tool/close checks required selected reruns after provider failures.
Functional reuse across sequential event loops also worked; this does not prove
leak-free cleanup after a loop has already closed. Service applications should
use one event loop and an async context manager.

The initial complete 28-case run without retries had 24 passes and four Google
failures: HTTP 500, HTTP 503, another HTTP 500 and an answerless stream. Four
additional timeout/cancellation checks passed. Reruns used three retries; normal
output allowance was subsequently increased from 1024 to 4096 for reasoning
headroom. This is not an all-green live run or a provider availability benchmark.

## SDK defect found and fixed

A Google async stream returned a single chunk containing 36 text characters,
no finish reason, and no completed answer. Reported usage was 7 prompt tokens,
0 completion tokens and 15 total tokens. Before the fix, the SDK returned an
empty answer without a stop reason.

Chat and Responses streams now raise `ModelRequestError` when they end without
usable answer text, a completed tool call, refusal or terminal status. The
Google failure reproduced after the fix and correctly raised that error instead
of reporting an empty success. Already-consumed events are not replayed.

Forty HTTP/SSE regression cases cover empty bodies, DONE-only streams,
unfinished reasoning, whitespace-only streams and deliberately completed empty
answers across sync/async, streaming/aggregated and Chat/Responses paths.
Explicitly completed empty answers and reported token-limit termination retain
their normal result semantics. Nonempty partial output can still have no stop
reason; applications must check termination before committing business results.

## Endpoint limitations that remain visible

### Responses proxy ignores output limits

With `max_output_tokens=16`, the proxy returned a completed response containing
1510 answer characters and **387 output tokens**, with no incomplete reason.
This was reproduced directly with the official OpenAI client, without llm-sdk.
The SDK serializes the correct field; it cannot make this proxy honor it.

Do not rely on this proxy for bounded output/cost until its output-limit behavior
is corrected upstream. The live truncation test intentionally fails here.

### Google failures, reasoning configuration and unfinished streams

Google intermittently returned HTTP 500 or HTTP 503 (`UNAVAILABLE`, high demand)
and sometimes ended a stream before an answer/terminal status. Request errors
were wrapped as `ModelRequestError`; the newly covered unfinished case is now
wrapped too. Retries do not guarantee recovery, especially after stream events
have already been consumed.

Google also returned HTTP 500 for the 16-token truncation probe. The same request
failed directly through the official OpenAI client, without llm-sdk. With
`reasoning_effort="minimal"` and three retries, the Google async-stream and
16-token truncation checks both passed; truncation was reported as `length`.
This is configuration-sensitive provider behavior, not evidence of a client-side
token-limit mapping bug. Do not report the automatic-reasoning probe as passing.

For the demonstrated Google profile, set `LLM_LIVE_CHAT_REASONING_EFFORT=minimal`
and `LLM_LIVE_MAX_RETRIES=3`. This passes an explicit caller option through the
existing API; no Google-specific normalization or automatic fallback was added.

## Reproduction and boundaries

See COMPATIBILITY.md for the 34 opt-in live cases and their environment settings.
Normal CI runs the deterministic protocol tests and skips live requests. Live
failures are kept visible; provider-specific xfails or automatic model fallbacks
have not been added to make the results green.

The core inference interface has stronger evidence after these checks. The tested
Responses proxy's output limits and Google's observed stream reliability prevent
claiming that these exact deployments are ready for every startup workload.
PDF/audio/video inputs, sustained load and other providers were not live-tested.
No routing, agent loop, pricing database or additional runtime dependency was added.

"""Opt-in live contracts. Credentials come only from environment variables."""

import asyncio
import os
from contextlib import aclosing, closing
from dataclasses import replace

import pytest
from openai import APITimeoutError
from pydantic import BaseModel, Field

from llm_sdk import LLM, LLMConfig, ModelRequestError, assistant_message, tool_result, user_message

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("LLM_SDK_LIVE") != "1", reason="set LLM_SDK_LIVE=1 to run live contracts"),
]


class Calculation(BaseModel):
    result: int = Field(alias="answer", ge=0, le=100)


@pytest.fixture(params=["chat", "responses"])
def endpoint(request):
    prefix = "LLM_LIVE_" + request.param.upper() + "_"
    for name in ("MODEL", "BASE_URL", "API_KEY"):
        if not os.getenv(prefix + name):
            pytest.fail(f"Missing live configuration: {prefix + name}")
    return LLMConfig(
        model=os.environ[prefix + "MODEL"],
        base_url=os.environ[prefix + "BASE_URL"],
        api_key=os.environ[prefix + "API_KEY"],
        use_responses_api=request.param == "responses",
        timeout=90,
        max_retries=int(os.getenv("LLM_LIVE_MAX_RETRIES", "0")),
    )


@pytest.fixture
def options(endpoint):
    prefix = "LLM_LIVE_" + ("RESPONSES" if endpoint.use_responses_api else "CHAT") + "_"
    result = {"max_tokens": 4096}
    effort = os.getenv(prefix + "REASONING_EFFORT")
    if effort:
        result["reasoning_effort"] = effort
    if endpoint.use_responses_api:
        result["store"] = False
    return result


@pytest.fixture
def llm(endpoint):
    with LLM.from_config(endpoint) as client:
        yield client


def check_result(result):
    assert isinstance(result["answer"], str) and result["answer"].strip(), (
        f"Empty answer: stop_reason={result.get('stop_reason')!r}, "
        f"reasoning_unterminated={result.get('reasoning_unterminated', False)!r}, usage={result.get('usage')!r}"
    )
    assert result["stop_reason"] == "stop"
    usage = result["usage"]
    assert all(value is None or isinstance(value, int) and value >= 0 for value in usage.values())
    assert usage["total_tokens"] is not None


def check_stream(events):
    assert events[-1] == {"type": "done", "content": None}
    finals = [event["content"] for event in events if event["type"] == "final"]
    assert len(finals) == 1
    result = finals[0]
    check_result(result)
    assert "".join(event["content"] for event in events if event["type"] == "answer") == result["answer"]


def test_sync_text_and_system(llm, options):
    result = llm.response(system="Answer briefly in English.", input="What is six times seven?", **options)
    check_result(result)
    assert "42" in result["answer"]


def test_sync_stream(llm, options):
    with closing(llm.stream_response(input="List three primary colors briefly.", final=True, **options)) as stream:
        check_stream(list(stream))


def test_async_text(llm, options):
    async def run():
        async with llm:
            result = await llm.async_response(input="What is six times seven?", **options)
            check_result(result)
            assert "42" in result["answer"]
    asyncio.run(run())


def test_async_stream(llm, options):
    async def run():
        async with llm, aclosing(llm.async_stream_response(input="List three primary colors briefly.", final=True, **options)) as stream:
            check_stream([event async for event in stream])
    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_structured_output(llm, options, asynchronous):
    kwargs = {"input": "Calculate 6 * 7. Return the result using the required answer field.", "output_format": Calculation, **options}
    if asynchronous:
        async def run():
            async with llm:
                return await llm.async_response(**kwargs)
        result = asyncio.run(run())
    else:
        result = llm.response(**kwargs)
    assert result["answer"] == {"answer": 42}
    assert result["stop_reason"] == "stop"


def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tool_round_trip(llm, options, asynchronous):
    history = [user_message("Call the add tool with left=19 and right=23, then report its result.")]

    def append_tool_result(response):
        calls = response.get("tool_calls", [])
        assert len(calls) == 1
        call = calls[0]
        assert call["id"] and call["name"] == "add"
        assert call["arguments"] == {"left": 19, "right": 23}
        history.extend([assistant_message(response), tool_result(call, add(**call["arguments"]))])

    kwargs = {"messages": history, "tools": [add], "tool_choice": {"type": "function", "function": {"name": "add"}}, **options}
    if asynchronous:
        async def run():
            async with llm:
                append_tool_result(await llm.async_response(**kwargs))
                return await llm.async_response(messages=history, tools=[add], tool_choice="none", **options)
        result = asyncio.run(run())
    else:
        append_tool_result(llm.response(**kwargs))
        result = llm.response(messages=history, tools=[add], tool_choice="none", **options)
    check_result(result)
    assert "42" in result["answer"]


def test_image_input(llm, options):
    image = pytest.importorskip("PIL.Image")
    result = llm.response(messages=[user_message([
        {"type": "text", "text": "Which color fills this image? Answer with one English color name."},
        {"type": "image", "image_pil": image.new("RGB", (64, 64), (255, 0, 0))},
    ])], **options)
    check_result(result)
    assert "red" in result["answer"].lower()


def test_async_concurrent_requests_are_isolated(llm, options):
    async def run():
        async with llm:
            results = await asyncio.gather(*[
                llm.async_response(input=f"Reply only with the word {word}.", **options)
                for word in ("ALPHA", "BRAVO", "CHARLIE")
            ])
            for word, result in zip(("ALPHA", "BRAVO", "CHARLIE"), results, strict=True):
                check_result(result)
                assert word in result["answer"].upper()
    asyncio.run(run())


def test_repeated_event_loops_reuse_instance(llm, options):
    for _ in range(2):
        result = asyncio.run(llm.async_response(input="Reply with READY.", **options))
        check_result(result)
        assert "READY" in result["answer"]
    asyncio.run(llm.aclose())


def test_sync_early_close_then_reuse(llm, options):
    termination = None
    with closing(llm.stream_response(input="Count from one to one hundred.", final=True, **options)) as stream:
        for event in stream:
            if event["type"] == "final":
                termination = event["content"].get("stop_reason")
            if event["type"] == "answer":
                break
        else:
            pytest.fail(f"Stream did not emit answer text (stop_reason={termination!r})")
    check_result(llm.response(input="Reply with READY.", **options))


def test_async_early_close_then_reuse(llm, options):
    async def run():
        async with llm:
            termination = None
            async with aclosing(llm.async_stream_response(input="Count from one to one hundred.", final=True, **options)) as stream:
                async for event in stream:
                    if event["type"] == "final":
                        termination = event["content"].get("stop_reason")
                    if event["type"] == "answer":
                        break
                else:
                    pytest.fail(f"Stream did not emit answer text (stop_reason={termination!r})")
            check_result(await llm.async_response(input="Reply with READY.", **options))
    asyncio.run(run())


def test_invalid_credentials_raise_typed_error(endpoint, options):
    with LLM.from_config(replace(endpoint, api_key="llm-sdk-invalid-test-key")) as client, pytest.raises(ModelRequestError) as error:
        client.response(input="Reply with READY.", **options)
    assert error.value.status_code in (400, 401, 403)


def test_timeout_raises_typed_error(endpoint, options):
    with LLM.from_config(replace(endpoint, timeout=0.001)) as client, pytest.raises(ModelRequestError) as error:
        client.response(input="Reply with READY.", **options)
    assert error.value.status_code is None
    assert isinstance(error.value.__cause__, APITimeoutError)


def test_output_limit_is_reported_as_truncation(llm, options):
    result = llm.response(
        input="Write at least two hundred words explaining prime numbers, with several examples.",
        **{**options, "max_tokens": 16},
    )
    assert result["stop_reason"] == "length"


def test_async_cancellation_then_reuse(llm, options):
    async def run():
        async with llm:
            pending = asyncio.create_task(llm.async_response(input="Count from one to one hundred.", **options))
            await asyncio.sleep(0.05)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            check_result(await llm.async_response(input="Reply with READY.", **options))
    asyncio.run(run())

"""Exercise serialization and SSE parsing through real OpenAI/HTTPX clients."""

import asyncio
import copy
import json
from types import SimpleNamespace
from typing import Annotated, Literal

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

import llm_sdk as sdk


class Verdict(BaseModel):
    score: int = Field(gt=0, alias="rating")


class Detail(BaseModel):
    verdict: Verdict


class Node(BaseModel):
    child: "Node | None" = None


class MapOutput(BaseModel):
    meta: dict[str, str]


class ExtraOutput(BaseModel):
    model_config = {"extra": "allow"}
    value: str


class NestedMapOutput(BaseModel):
    nested: MapOutput


def test_pydantic_refs_are_resolved_when_embedded_in_plain_classes():
    class Output:
        detail: Detail

    schema = sdk.SchemaConverter().convert_class_to_schema(Output)["json_schema"]["schema"]
    leaf = schema["properties"]["detail"]["properties"]["verdict"]
    assert leaf["properties"]["rating"]["exclusiveMinimum"] == 0
    assert '"$ref"' not in json.dumps(schema)


def test_pydantic_refs_are_resolved_in_tool_parameters():
    def inspect_detail(detail: Detail) -> str:
        return "ok"

    prepared = sdk.ToolPreparator(sdk.SchemaConverter()).prepare([inspect_detail])
    schema = prepared.definitions[0]["function"]["parameters"]
    assert schema["properties"]["detail"]["properties"]["verdict"]["required"] == ["rating"]
    assert '"$ref"' not in json.dumps(schema)


def test_embedded_aliases_and_example_data_are_not_schema_keywords():
    class Leaf(BaseModel):
        reference: str = Field(alias="$ref")
        definitions: str = Field(alias="$defs", json_schema_extra={"examples": [{"$ref": "literal"}]})

    # create_model resolves the local type without relying on local annotations.
    from pydantic import create_model

    nested = create_model("AliasedNested", leaf=(Leaf, ...))
    output = type("Output", (), {"__annotations__": {"nested": nested}})
    schema = sdk.SchemaConverter().convert_class_to_schema(output)["json_schema"]["schema"]
    leaf = schema["properties"]["nested"]["properties"]["leaf"]
    assert set(leaf["properties"]) == {"$ref", "$defs"}
    assert leaf["properties"]["$defs"]["examples"] == [{"$ref": "literal"}]


def test_recursive_pydantic_is_supported_at_root_but_rejected_when_embedded():
    converter = sdk.SchemaConverter()
    assert "$defs" in converter.convert_class_to_schema(Node)["json_schema"]["schema"]

    class Output:
        node: Node

    with pytest.raises(sdk.SchemaConversionError, match="Recursive schemas"):
        converter.convert_class_to_schema(Output)


def test_embedded_discriminated_union_does_not_keep_dangling_mapping_refs():
    class Cat(BaseModel):
        kind: Literal["cat"]

    class Dog(BaseModel):
        kind: Literal["dog"]

    class Zoo(BaseModel):
        pet: Annotated[Cat | Dog, Field(discriminator="kind")]

    class Output:
        zoo: Zoo

    schema = sdk.SchemaConverter().convert_class_to_schema(Output)["json_schema"]["schema"]
    variants = schema["properties"]["zoo"]["properties"]["pet"]["oneOf"]
    assert [v["properties"]["kind"]["const"] for v in variants] == ["cat", "dog"]
    assert '"discriminator"' not in json.dumps(schema)
    assert '"#/$defs/' not in json.dumps(schema)


def sse(responses, text='{"rating":2}', usage=True):
    if responses:
        events = [
            {"type": "response.output_text.delta", "delta": text},
            {"type": "response.completed", "response": {
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10} if usage else None,
            }},
        ]
    else:
        events = [
            {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        if usage:
            events.append({"choices": [], "usage": {
                "prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10,
            }})
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


@pytest.fixture
def server(monkeypatch):
    state = {"body": "", "requests": []}

    def handle(request):
        state["requests"].append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=state["body"])

    monkeypatch.setattr(sdk, "OpenAI", lambda **kwargs: openai.OpenAI(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ))
    monkeypatch.setattr(sdk, "AsyncOpenAI", lambda **kwargs: openai.AsyncOpenAI(
        **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ))
    return state


def request_model(responses, asynchronous, stream, **kwargs):
    if asynchronous:
        async def run():
            async with sdk.LLM(model="gpt-5-local", use_responses_api=responses) as llm:
                if stream:
                    return [event async for event in llm.async_stream_response(input="hi", final=True, **kwargs)]
                return await llm.async_response(input="hi", **kwargs)
        return asyncio.run(run())
    with sdk.LLM(model="gpt-5-local", use_responses_api=responses) as llm:
        if stream:
            return list(llm.stream_response(input="hi", final=True, **kwargs))
        return llm.response(input="hi", **kwargs)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", [MapOutput, ExtraOutput, NestedMapOutput])
def test_invalid_strict_pydantic_schemas_fail_before_network(server, responses, asynchronous, stream, model):
    with pytest.raises(sdk.SchemaConversionError, match="additionalProperties: false"):
        request_model(responses, asynchronous, stream, output_format=model)
    assert server["requests"] == []


@pytest.mark.parametrize("model", [MapOutput, ExtraOutput, NestedMapOutput])
def test_non_strict_pydantic_keeps_open_object_schemas(model):
    converted = sdk.SchemaConverter().convert_class_to_schema(model, strict=False)
    assert converted["json_schema"]["strict"] is False
    assert "additionalProperties" in json.dumps(converted)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_raw_strict_combinations_reject_open_objects(keyword):
    schema = {keyword: [{"type": "object", "additionalProperties": {"type": "string"}}]}
    with pytest.raises(sdk.SchemaConversionError, match="additionalProperties: false"):
        sdk.SchemaConverter._validate_strict_schema(schema)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_validated_output_and_usage_on_every_path(server, responses, asynchronous, stream):
    server["body"] = sse(responses)
    result = request_model(responses, asynchronous, stream, output_format=Verdict, max_tokens=64)
    if stream:
        assert [e["content"] for e in result if e["type"] == "answer"] == [{"rating": 2}]
        result = next(e["content"] for e in result if e["type"] == "final")
    assert result["answer"] == {"rating": 2}
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
    assert "verbose" not in result
    path, body = server["requests"][0]
    assert path == ("/v1/responses" if responses else "/v1/chat/completions")
    assert body["max_output_tokens" if responses else "max_tokens"] == 64
    schema = body["text"]["format"]["schema"] if responses else body["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["rating"]
    assert schema["properties"]["rating"]["exclusiveMinimum"] == 0


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ['{"rating":"2"}', '{"rating":0}', '{}', 'not json'])
def test_invalid_output_never_reaches_caller(server, responses, asynchronous, stream, text):
    server["body"] = sse(responses, text)
    with pytest.raises(sdk.StructuredOutputError) as error:
        request_model(responses, asynchronous, stream, output_format=Verdict)
    assert error.value.raw == text


@pytest.mark.parametrize("responses", [False, True])
def test_missing_usage_is_unknown_not_a_chunk_estimate(server, responses):
    server["body"] = sse(responses, "hello", usage=False)
    result = request_model(responses, False, False)
    assert result["usage"] == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("body", ["empty", "done_only", "unfinished_reasoning", "whitespace"])
def test_empty_or_reasoning_only_unfinished_stream_raises(server, responses, asynchronous, stream, body):
    if body == "empty":
        server["body"] = ""
    elif body == "done_only":
        server["body"] = "data: [DONE]\n\n"
    else:
        text = "   " if body == "whitespace" else "<think>unfinished"
        event = {"type": "response.output_text.delta", "delta": text} if responses else {
            "choices": [{"delta": {"content": text}}],
        }
        server["body"] = f"data: {json.dumps(event)}\n\n"
    with pytest.raises(sdk.ModelRequestError, match="stream ended before"):
        request_model(responses, asynchronous, stream)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_explicitly_completed_empty_answer_is_preserved(server, responses, asynchronous, stream):
    server["body"] = sse(responses, "")
    result = request_model(responses, asynchronous, stream)
    if stream:
        result = next(e["content"] for e in result if e["type"] == "final")
    assert result["answer"] == ""
    assert result["stop_reason"] == "stop"


def test_store_false_honors_explicit_includes_on_wire(server):
    server["body"] = sse(True, "hello")
    request_model(True, False, False, store=False, extra_body={"include": []})
    assert server["requests"][0][1]["include"] == []


def test_partial_usage_never_adds_chunks_to_real_tokens():
    assert sdk._resolve_token_metrics(None, 3, None, 2) == (2, None)


@pytest.mark.parametrize("part", [
    {"file_data": "data:text/plain;base64,aG k=\n", "filename": "hello.txt"},
    {"file_id": "file-123"},
    {"file_url": "https://example.com/a.pdf"},
])
def test_wire_files_are_validated_canonical_and_idempotent(part):
    item = {"type": "file", "file": part}
    original = copy.deepcopy(item)
    converted = sdk.FileProcessor._convert_file_item(item)
    assert item == original
    assert sdk.FileProcessor._convert_file_item(converted) == converted
    translated = sdk.LLM._translate_content_for_responses_api([converted])[0]
    assert translated["type"] == "input_file"
    if "file_data" in part:
        assert translated["file_data"] == "data:text/plain;base64,aGk="


@pytest.mark.parametrize("part", [
    {"file_data": "!!!"},
    {"file_url": "https://user:password@example.com/a.pdf"},
    {"file_id": ""},
    {"file_id": "file-123", "file_data": "aGk="},
    {"file_data": "aGk=", "filename": "bad\x00name"},
])
def test_invalid_wire_files_fail_in_both_modes(part):
    item = {"type": "file", "file": part}
    with pytest.raises(sdk.ConfigurationError):
        sdk.FileProcessor._convert_file_item(item)
    with pytest.raises(sdk.ConfigurationError):
        sdk.LLM._translate_content_for_responses_api([item])


@pytest.mark.parametrize("message", [
    {"role": "tool", "content": "ok"},
    {"role": "tool", "tool_call_id": " ", "content": "ok"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "test"}}]},
    {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {}}]},
])
def test_invalid_tool_history_fails_before_request(message):
    with sdk.LLM(model="local", use_responses_api=True) as llm, pytest.raises(sdk.ConfigurationError):
        llm.response(messages=[message])


def test_pillow_limit_is_preserved_and_send_budget_enforced(monkeypatch):
    image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(sdk.ImageProcessor, "_pil_image", None)
    old_limit = image.MAX_IMAGE_PIXELS
    sdk.ImageProcessor._get_pil()
    assert old_limit == image.MAX_IMAGE_PIXELS
    monkeypatch.setattr(sdk.ImageProcessor, "_MAX_IMAGE_SEND_BYTES", 1)
    with pytest.raises(sdk.ImageProcessingError):
        sdk.ImageProcessor._from_pil(image.new("RGB", (1, 1)))


def test_different_tools_without_ids_do_not_merge_incomplete_arguments():
    handler = sdk.ToolCallStreamHandler(sdk.EventBuilder())
    handler.process_chunk([SimpleNamespace(function=SimpleNamespace(name="first", arguments='{"a":'))])
    handler.process_chunk([SimpleNamespace(function=SimpleNamespace(name="second", arguments='{"b":2}'))])
    calls = handler.finalize()
    assert [(c["name"], c["arguments"]) for c in calls] == [
        ("first", {"_raw": '{"a":'}), ("second", {"b": 2}),
    ]


def test_constant_tool_index_without_ids_splits_names_and_redirects_later_deltas():
    handler = sdk.ToolCallStreamHandler(sdk.EventBuilder())
    handler.process_chunk([SimpleNamespace(index=0, function=SimpleNamespace(name="first", arguments='{"a":1}'))])
    handler.process_chunk([SimpleNamespace(index=0, function=SimpleNamespace(name="second", arguments='{"b":'))])
    handler.process_chunk([SimpleNamespace(index=0, function=SimpleNamespace(name=None, arguments='2}'))])
    handler.finalize()
    calls = handler.get_all_calls()
    assert [(c["name"], c["arguments"]) for c in calls] == [
        ("first", {"a": 1}), ("second", {"b": 2}),
    ]
    assert calls[0]["id"] != calls[1]["id"]


def test_responses_idless_items_keep_their_own_arguments_and_call_ids():
    with sdk.LLM(model="local") as llm:
        state = llm._new_responses_state()
        for index, name in enumerate(("first", "second")):
            llm._handle_responses_event(SimpleNamespace(
                type="response.output_item.added", output_index=index,
                item=SimpleNamespace(type="function_call", call_id=f"call_{index}", name=name),
            ), state, False, True, {})
            llm._handle_responses_event(SimpleNamespace(
                type="response.function_call_arguments.delta", output_index=index, delta='{"n":' + str(index) + '}',
            ), state, False, True, {})
        for index in range(2):
            llm._handle_responses_event(SimpleNamespace(
                type="response.output_item.done", output_index=index,
                item=SimpleNamespace(type="function_call"),
            ), state, False, True, {})
        assert [(c["id"], c["name"], c["arguments"]) for c in state["completed_tool_calls"]] == [
            ("call_0", "first", {"n": 0}), ("call_1", "second", {"n": 1}),
        ]


def test_async_responses_early_close_propagates(monkeypatch):
    closed = []

    async def events(self, **kwargs):
        try:
            yield {"type": "answer", "content": "hello"}
            yield {"type": "done", "content": None}
        finally:
            closed.append(True)

    monkeypatch.setattr(sdk.LLM, "_stream_responses_async", events)

    async def run():
        async with sdk.LLM(model="local", use_responses_api=True) as llm:
            stream = llm.async_stream_response(input="hi")
            await anext(stream)
            await stream.aclose()
            assert closed == [True]
    asyncio.run(run())

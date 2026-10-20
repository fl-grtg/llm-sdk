import os

from llm_sdk import LLM, assistant_message, tool_result, user_message


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


functions = {"celsius_to_fahrenheit": celsius_to_fahrenheit}
messages = [user_message("Use the conversion tool to convert 21 degrees Celsius to Fahrenheit.")]

with LLM(
    model=os.environ["LLM_MODEL"],
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LLM_API_KEY", "lm-studio"),
) as llm:
    response = llm.response(messages=messages, tools=list(functions.values()))
    if response.get("tool_calls"):
        messages.append(assistant_message(response))
        for call in response["tool_calls"]:
            function = functions[call["name"]]
            messages.append(tool_result(call, function(**call["arguments"])))
        response = llm.response(messages=messages, tools=list(functions.values()), tool_choice="none")
    print(response["answer"])

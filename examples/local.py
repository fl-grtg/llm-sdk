import os

from llm_sdk import LLM

with LLM(
    model=os.environ["LLM_MODEL"],
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LLM_API_KEY", "lm-studio"),
) as llm:
    response = llm.response(input="Write a tiny haiku about fast code.")
    print(response["answer"])
    print(response["usage"])

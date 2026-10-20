import os
from contextlib import closing

from llm_sdk import LLM

with LLM(
    model=os.environ["LLM_MODEL"],
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LLM_API_KEY", "lm-studio"),
) as llm:
    with closing(llm.stream_response(input="Explain streaming in two sentences.")) as stream:
        for event in stream:
            if event["type"] == "answer":
                print(event["content"], end="", flush=True)
    print()

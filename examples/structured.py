import os

from pydantic import BaseModel, Field

from llm_sdk import LLM


class Review(BaseModel):
    sentiment: str
    score: int = Field(ge=1, le=5)


with LLM(
    model=os.environ["LLM_MODEL"],
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LLM_API_KEY", "lm-studio"),
) as llm:
    response = llm.response(input="Review: fast, small, surprisingly nice.", output_format=Review)
    # The SDK validates the model, and keeps the answer as JSON-compatible data.
    print(response["answer"])

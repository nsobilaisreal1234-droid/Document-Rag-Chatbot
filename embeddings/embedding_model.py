import json

import httpx
from langchain_core.embeddings import Embeddings

from app.config import settings


def _openrouter_embedding_request(
    texts: list[str],
    api_key: str,
    model: str,
    dimensions: int,
) -> dict:
    payload = {
        "model": model,
        "input": texts,
        "dimensions": dimensions,
    }

    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.post(
                f"{settings.OPENROUTER_BASE_URL}/embeddings",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "Document-RAG-Chatbot/1.0",
                    "X-Title": "Document RAG Chatbot",
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise RuntimeError(
            "OpenRouter embeddings request failed "
            f"({exc.response.status_code}): {detail}"
        ) from exc
    except (httpx.RequestError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenRouter embeddings request failed: {exc}") from exc


class OpenRouterGeminiEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = settings.OPENROUTER_EMBEDDING_MODEL,
        dimensions: int = settings.OPENROUTER_EMBEDDING_DIMENSIONS,
    ) -> None:
        self.api_key = api_key or settings.get_setting("OPENROUTER_API_KEY")
        self.model = model
        self.dimensions = dimensions

        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = _openrouter_embedding_request(
            texts=texts,
            api_key=self.api_key,
            model=self.model,
            dimensions=self.dimensions,
        )
        return [item["embedding"] for item in response["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding(texts, api_key: str | None = None):
    embeddings = OpenRouterGeminiEmbeddings(api_key=api_key)
    if isinstance(texts, str):
        return [embeddings.embed_query(texts)]

    return embeddings.embed_documents(texts)

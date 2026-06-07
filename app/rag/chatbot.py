import json
from dataclasses import dataclass

import httpx
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config.settings import (
    DEFAULT_GROQ_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_CHAT_MODEL,
    get_setting,
)
from app.vectorstore.supabase_store import retrieve_chunks


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer ONLY using the provided context. "
                "The context may include document metadata, the document "
                "profile, detected sections, notes, key info, header, "
                "and retrieved chunks. "
                "If the answer is not in the context, say you don't know."
            ),
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass(frozen=True)
class RagAnswer:
    content: str
    provider: str
    model: str
    fallback_reason: str | None = None


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _openrouter_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    role_by_type = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }

    return [
        {
            "role": role_by_type.get(message.type, message.type),
            "content": _message_content_to_text(message.content),
        }
        for message in messages
    ]


def _format_rag_messages(question: str, context: str) -> list[BaseMessage]:
    return RAG_PROMPT.invoke(
        {
            "question": question,
            "context": context,
        }
    ).to_messages()


def _groq_chat_request(
    messages: list[BaseMessage],
    api_key: str,
    model: str,
    temperature: float = 0.2,
) -> str:
    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=60,
        max_retries=1,
    )

    response = llm.invoke(messages)
    return _message_content_to_text(response.content)


def _openrouter_chat_request(
    messages: list[BaseMessage],
    api_key: str,
    model: str = OPENROUTER_CHAT_MODEL,
    temperature: float = 0.2,
) -> str:
    payload = {
        "model": model,
        "messages": _openrouter_messages(messages),
        "temperature": temperature,
    }

    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
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
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise RuntimeError(
            "OpenRouter chat fallback failed "
            f"({exc.response.status_code}): {detail}"
        ) from exc
    except (httpx.RequestError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenRouter chat fallback failed: {exc}") from exc


def _short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    return message[:220] + "..." if len(message) > 220 else message


def get_rag_answer(
    question: str,
    context: str,
    groq_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    model: str = DEFAULT_GROQ_MODEL,
) -> RagAnswer:
    resolved_groq_api_key = groq_api_key or get_setting("GROQ_API_KEY")
    resolved_openrouter_api_key = (
        openrouter_api_key or get_setting("OPENROUTER_API_KEY")
    )

    if not resolved_groq_api_key:
        raise ValueError("Missing GROQ_API_KEY.")

    messages = _format_rag_messages(question, context)

    try:
        return RagAnswer(
            content=_groq_chat_request(
                messages=messages,
                api_key=resolved_groq_api_key,
                model=model,
            ),
            provider="Groq",
            model=model,
        )
    except Exception as groq_exc:
        if not resolved_openrouter_api_key:
            raise RuntimeError(
                "Groq chat failed and OPENROUTER_API_KEY is missing, "
                "so fallback chat could not run."
            ) from groq_exc

        try:
            return RagAnswer(
                content=_openrouter_chat_request(
                    messages=messages,
                    api_key=resolved_openrouter_api_key,
                ),
                provider="OpenRouter fallback",
                model=OPENROUTER_CHAT_MODEL,
                fallback_reason=_short_error(groq_exc),
            )
        except Exception as openrouter_exc:
            raise RuntimeError(
                "Groq chat failed and OpenRouter fallback also failed. "
                f"Groq: {_short_error(groq_exc)} "
                f"OpenRouter: {_short_error(openrouter_exc)}"
            ) from openrouter_exc


def get_groq_answer(
    question: str,
    context: str,
    api_key: str | None = None,
    model: str = DEFAULT_GROQ_MODEL,
) -> str:
    return get_rag_answer(
        question=question,
        context=context,
        groq_api_key=api_key,
        model=model,
    ).content


def ask_chatbot(question, top_k=5, model=DEFAULT_GROQ_MODEL):
    retrieved_docs = retrieve_chunks(question, top_k=top_k)
    context = "\n".join(doc["content"] for doc in retrieved_docs)

    return get_groq_answer(question, context, model=model)

import importlib
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client

import app.config.settings as app_settings
from app.rag.chatbot import get_rag_answer
from embeddings.embedding_model import get_embedding


app_settings = importlib.reload(app_settings)

DEFAULT_GROQ_MODEL = app_settings.DEFAULT_GROQ_MODEL
ENV_FILE = app_settings.ENV_FILE
GROQ_MODELS = app_settings.GROQ_MODELS
REQUIRED_ENV_VARS = app_settings.REQUIRED_ENV_VARS
apply_runtime_settings = app_settings.apply_runtime_settings
get_runtime_settings = app_settings.get_runtime_settings
missing_required_settings = app_settings.missing_required_settings
save_settings_to_env = app_settings.save_settings_to_env

SETTINGS_PREFIX = "credential_"
APP_SESSION_VERSION = "document-profile-v3"
EMBEDDING_BATCH_SIZE = 24
INSERT_BATCH_SIZE = 100


st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.app-kicker {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.app-title {
    color: #111827;
    font-size: 2rem;
    font-weight: 750;
    line-height: 1.15;
    margin: 0;
}

.app-subtitle {
    color: #475569;
    margin-top: 0.45rem;
    margin-bottom: 1rem;
}

.status-card {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-left: 4px solid #64748b;
    border-radius: 8px;
    min-height: 88px;
    padding: 0.8rem 0.9rem;
}

.status-card.ready {
    border-left-color: #16a34a;
}

.status-card.warn {
    border-left-color: #f59e0b;
}

.status-card.error {
    border-left-color: #dc2626;
}

.status-card.info {
    border-left-color: #2563eb;
}

.status-label {
    color: #64748b;
    display: block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
}

.status-value {
    color: #111827;
    display: block;
    font-size: 1rem;
    font-weight: 750;
    margin-top: 0.25rem;
}

.status-detail {
    color: #64748b;
    display: block;
    font-size: 0.84rem;
    margin-top: 0.2rem;
}

.sidebar-badge {
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    margin-bottom: 0.75rem;
    padding: 0.65rem 0.75rem;
}

.sidebar-badge.ready {
    background: #f0fdf4;
    border-color: #bbf7d0;
    color: #166534;
}

.sidebar-badge.warn {
    background: #fffbeb;
    border-color: #fde68a;
    color: #92400e;
}

.sidebar-badge.error {
    background: #fef2f2;
    border-color: #fecaca;
    color: #991b1b;
}

.sidebar-badge-title {
    display: block;
    font-size: 0.82rem;
    font-weight: 750;
}

.sidebar-badge-detail {
    display: block;
    font-size: 0.78rem;
    margin-top: 0.1rem;
}

.chat-empty {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    color: #475569;
    margin-top: 0.5rem;
    padding: 1.1rem 1.2rem;
}

.chat-empty strong {
    color: #111827;
    display: block;
    margin-bottom: 0.2rem;
}

.chat-msg {
    border-radius: 8px;
    margin-bottom: 0.45rem;
    padding: 0.75rem 0.9rem;
}

.chat-msg.user {
    background: #f1f5f9;
    border: 1px solid #d9e2ec;
}

.chat-msg.assistant {
    background: #eef6ff;
    border: 1px solid #bfdbfe;
}

.chat-time {
    color: #94a3b8;
    display: block;
    font-size: 0.76rem;
    margin-top: 0.2rem;
}

.provider-tag {
    background: #f8fafc;
    border: 1px solid #d9e2ec;
    border-radius: 999px;
    color: #475569;
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 0.1rem 0 0.3rem;
    padding: 0.16rem 0.5rem;
}

.stChatMessage {
    padding: 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def initialize_credential_state() -> None:
    env_settings = get_runtime_settings()

    for name in REQUIRED_ENV_VARS:
        st.session_state.setdefault(
            f"{SETTINGS_PREFIX}{name}",
            env_settings.get(name, ""),
        )


def get_credentials_from_state() -> dict[str, str]:
    return {
        name: st.session_state.get(f"{SETTINGS_PREFIX}{name}", "").strip()
        for name in REQUIRED_ENV_VARS
    }


def render_status_card(
    label: str,
    value: str,
    detail: str,
    tone: str,
) -> None:
    st.markdown(
        f"""
<div class="status-card {tone}">
    <span class="status-label">{escape(label)}</span>
    <span class="status-value">{escape(value)}</span>
    <span class="status-detail">{escape(detail)}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_badge(title: str, detail: str, tone: str) -> None:
    st.markdown(
        f"""
<div class="sidebar-badge {tone}">
    <span class="sidebar-badge-title">{escape(title)}</span>
    <span class="sidebar-badge-detail">{escape(detail)}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_settings(settings_ready: bool) -> dict[str, str]:
    with st.expander("Settings", expanded=not settings_ready):
        with st.form("credential_settings"):
            st.text_input(
                "GROQ_API_KEY",
                type="password",
                key=f"{SETTINGS_PREFIX}GROQ_API_KEY",
            )
            st.text_input(
                "OPENROUTER_API_KEY",
                type="password",
                key=f"{SETTINGS_PREFIX}OPENROUTER_API_KEY",
            )
            st.text_input(
                "SUPABASE_URL",
                key=f"{SETTINGS_PREFIX}SUPABASE_URL",
            )
            st.text_input(
                "SUPABASE_KEY",
                type="password",
                key=f"{SETTINGS_PREFIX}SUPABASE_KEY",
            )

            save_to_env = st.checkbox("Save to .env")
            submitted = st.form_submit_button(
                "Apply Settings",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            credentials = get_credentials_from_state()
            apply_runtime_settings(credentials)

            if save_to_env:
                save_settings_to_env(credentials)
                st.success(f"Saved settings to {ENV_FILE.name}.")
            else:
                st.success("Settings applied for this session.")

            st.rerun()

    return get_credentials_from_state()


def get_supabase_client(credentials: dict[str, str]):
    client_credentials = (
        credentials["SUPABASE_URL"],
        credentials["SUPABASE_KEY"],
    )

    if (
        "supabase_client" not in st.session_state
        or st.session_state.get("supabase_client_credentials")
        != client_credentials
    ):
        st.session_state.supabase_client = create_client(
            credentials["SUPABASE_URL"],
            credentials["SUPABASE_KEY"],
        )
        st.session_state.supabase_client_credentials = client_credentials

    return st.session_state.supabase_client


def load_uploaded_documents(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        return loader.load()
    finally:
        os.unlink(tmp_path)


def batched(items: list, batch_size: int) -> Iterable[list]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings = []

    for batch in batched(chunks, EMBEDDING_BATCH_SIZE):
        embeddings.extend(get_embedding(batch))

    return embeddings


def insert_document_rows(supabase, rows: list[dict]) -> None:
    for batch in batched(rows, INSERT_BATCH_SIZE):
        supabase.table("documents").insert(batch).execute()


def clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


KNOWN_SECTION_HEADINGS = {
    "summary",
    "profile",
    "professional summary",
    "experience",
    "work experience",
    "employment history",
    "projects",
    "education",
    "certifications",
    "certification",
    "skills",
    "technical skills",
    "tools",
    "technologies",
    "awards",
    "publications",
    "contact",
    "languages",
    "references",
}


def is_likely_section_heading(line: str) -> bool:
    normalized = re.sub(r"[:\-]+$", "", line.strip()).lower()

    if normalized in KNOWN_SECTION_HEADINGS:
        return True

    if len(line) > 70 or len(line.split()) > 8:
        return False

    if re.search(r"@|https?://|www\.|\d{3,}", line):
        return False

    if line.startswith(("-", "•", "*")) or line.endswith("."):
        return False

    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    return uppercase_ratio > 0.65 or line.istitle()


def extract_unique_matches(pattern: str, text: str, limit: int = 8) -> list[str]:
    matches = []
    seen = set()

    for match in re.findall(pattern, text):
        value = match.strip(" .,:;()[]")
        if value and value not in seen:
            matches.append(value)
            seen.add(value)

        if len(matches) >= limit:
            break

    return matches


def build_document_profile(text: str, filename: str) -> dict:
    lines = clean_lines(text)
    header_lines = lines[:8]
    section_names = []

    for line in lines:
        if is_likely_section_heading(line):
            heading = re.sub(r"[:\-]+$", "", line.strip())
            if heading not in section_names:
                section_names.append(heading)

        if len(section_names) >= 16:
            break

    emails = extract_unique_matches(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text,
    )
    phones = extract_unique_matches(
        r"(?:\+?\d[\d\s().-]{7,}\d)",
        text,
        limit=6,
    )
    links = extract_unique_matches(
        r"(?:https?://|www\.)[^\s<>()]+",
        text,
        limit=8,
    )

    notes = [
        f"{len(lines)} non-empty lines extracted",
        f"{len(section_names)} likely sections detected",
    ]

    if emails or phones or links:
        notes.append("Contact details detected")

    if re.search(r"\s\|\s|(?:\n.*){0,2}\b(Table|Column|Row)\b", text, re.I):
        notes.append("Structured/table-like content detected")

    return {
        "title": header_lines[0] if header_lines else filename_words(filename),
        "header_lines": header_lines,
        "sections": section_names,
        "emails": emails,
        "phones": phones,
        "links": links,
        "notes": notes,
    }


def format_document_profile(profile: dict | None) -> str:
    if not profile:
        return "No document profile available."

    lines = [
        f"Title/header: {profile.get('title', 'Unknown')}",
    ]

    sections = profile.get("sections") or []
    if sections:
        lines.append("Detected sections: " + ", ".join(sections[:12]))

    emails = profile.get("emails") or []
    phones = profile.get("phones") or []
    links = profile.get("links") or []
    if emails:
        lines.append("Emails: " + ", ".join(emails))
    if phones:
        lines.append("Phones: " + ", ".join(phones))
    if links:
        lines.append("Links: " + ", ".join(links))

    notes = profile.get("notes") or []
    if notes:
        lines.append("Notes: " + "; ".join(notes))

    return "\n".join(lines)


def rank_fallback_chunks(question: str, chunks: list[str], top_k: int) -> list[str]:
    query_terms = set(re.findall(r"\w+", question.lower()))

    if not query_terms:
        return chunks[:top_k]

    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: len(
            query_terms.intersection(re.findall(r"\w+", chunk.lower()))
        ),
        reverse=True,
    )
    return ranked_chunks[:top_k]


def fetch_document_chunks(supabase, doc_info: dict) -> list[str]:
    query = supabase.table("documents").select("id, content").order("id")

    if doc_info.get("document_id"):
        query = query.contains(
            "metadata",
            {"document_id": doc_info["document_id"]},
        )
    else:
        query = query.contains("metadata", {"filename": doc_info["filename"]})

    response = query.limit(500).execute()
    return [row["content"] for row in response.data or []]


def filename_words(filename: str) -> str:
    return re.sub(r"[-_]+", " ", Path(filename).stem).strip()


def row_belongs_to_current_document(row: dict, doc_info: dict) -> bool:
    metadata = row.get("metadata") or {}

    if doc_info.get("document_id"):
        return metadata.get("document_id") == doc_info["document_id"]

    return metadata.get("filename") == doc_info["filename"]


def unique_chunks(chunks: list[str]) -> list[str]:
    seen = set()
    unique = []

    for chunk in chunks:
        normalized = " ".join(chunk.split())
        if normalized and normalized not in seen:
            unique.append(chunk)
            seen.add(normalized)

    return unique


def build_answer_context(
    doc_info: dict,
    document_chunks: list[str],
    matched_chunks: list[str],
    document_profile: dict | None = None,
) -> str:
    filename = str(doc_info["filename"])
    sections = [
        (
            "Document metadata:\n"
            f"Filename: {filename}\n"
            f"Filename words: {filename_words(filename)}"
        )
    ]

    if document_profile:
        sections.append(
            "Document profile, sections, notes, and key info:\n"
            + format_document_profile(document_profile)
        )

    if document_chunks:
        sections.append(
            "Document header / first chunk:\n" + document_chunks[0]
        )

    retrieved_context = unique_chunks(matched_chunks)
    if document_chunks:
        header_text = " ".join(document_chunks[0].split())
        retrieved_context = [
            chunk
            for chunk in retrieved_context
            if " ".join(chunk.split()) != header_text
        ]

    if retrieved_context:
        sections.append(
            "Retrieved relevant chunks:\n"
            + "\n\n".join(retrieved_context)
        )

    return "\n\n".join(sections)


def render_message(message: dict[str, str], show_sources: bool) -> None:
    role = message["role"]
    css_role = "user" if role == "user" else "assistant"

    with st.chat_message(role):
        safe_content = escape(message["content"]).replace("\n", "<br>")
        st.markdown(
            f"<div class='chat-msg {css_role}'>{safe_content}</div>",
            unsafe_allow_html=True,
        )

        if show_sources and role == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {escape(str(source))}")

        provider = message.get("provider")
        if role == "assistant" and provider:
            st.markdown(
                f"<span class='provider-tag'>{escape(provider)}</span>",
                unsafe_allow_html=True,
            )
            if message.get("fallback_reason"):
                st.caption(f"Fallback reason: {message['fallback_reason']}")

        st.markdown(
            f"<span class='chat-time'>{escape(message['time'])}</span>",
            unsafe_allow_html=True,
        )


initialize_credential_state()
credentials = get_credentials_from_state()
apply_runtime_settings(credentials)
missing_settings = missing_required_settings(credentials)
settings_ready = not missing_settings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "doc_profile" not in st.session_state:
    st.session_state.doc_profile = None

if st.session_state.get("app_session_version") != APP_SESSION_VERSION:
    st.session_state.messages = []
    st.session_state.app_session_version = APP_SESSION_VERSION

if not st.session_state.doc_info:
    st.session_state.doc_chunks = []
    st.session_state.doc_profile = None
    if st.session_state.messages:
        st.session_state.messages = []

if (
    st.session_state.doc_info
    and st.session_state.doc_chunks
    and not st.session_state.doc_profile
):
    st.session_state.doc_profile = build_document_profile(
        "\n".join(st.session_state.doc_chunks),
        st.session_state.doc_info["filename"],
    )

supabase = None
connection_error = None

if settings_ready:
    try:
        supabase = get_supabase_client(credentials)
    except Exception as exc:
        settings_ready = False
        connection_error = str(exc)

with st.sidebar:
    st.markdown("## RAG Chatbot")

    if settings_ready:
        render_sidebar_badge(
            "Settings ready",
            "Groq primary, OpenRouter fallback, Supabase ready.",
            "ready",
        )
    elif connection_error:
        render_sidebar_badge(
            "Connection issue",
            connection_error,
            "error",
        )
    else:
        render_sidebar_badge(
            "Settings incomplete",
            "Missing: " + ", ".join(missing_settings),
            "warn",
        )

    st.divider()

    st.markdown("### Document")

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt"],
        disabled=not settings_ready,
    )

    process_disabled = (
        not settings_ready
        or uploaded_file is None
        or st.session_state.doc_info is not None
    )
    process_clicked = st.button(
        "Process and Save",
        use_container_width=True,
        type="primary",
        disabled=process_disabled,
    )

    if uploaded_file and st.session_state.doc_info is not None:
        st.caption("Clear the current document before processing another one.")

    if (
        uploaded_file
        and st.session_state.doc_info is None
        and process_clicked
        and supabase is not None
    ):
        with st.spinner("Processing document..."):
            try:
                docs = load_uploaded_documents(uploaded_file)
                full_text = "\n".join(doc.page_content for doc in docs)
                chunks = text_splitter.split_text(full_text)
                document_id = str(uuid4())
                document_profile = build_document_profile(
                    full_text,
                    uploaded_file.name,
                )

                if not chunks:
                    st.error("No text chunks were created.")
                else:
                    embeddings = embed_chunks(chunks)

                    data = []

                    for chunk_index, (chunk, embedding) in enumerate(
                        zip(chunks, embeddings)
                    ):
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()

                        data.append(
                            {
                                "content": chunk,
                                "embedding": embedding,
                                "metadata": {
                                    "document_id": document_id,
                                    "filename": uploaded_file.name,
                                    "chunk_index": chunk_index,
                                },
                            }
                        )

                    insert_document_rows(supabase, data)

                    st.session_state.doc_info = {
                        "document_id": document_id,
                        "filename": uploaded_file.name,
                        "pages": len(docs),
                        "chunks": len(chunks),
                        "uploaded_at": datetime.now().strftime(
                            "%b %d, %Y %I:%M %p"
                        ),
                    }
                    st.session_state.doc_chunks = chunks
                    st.session_state.doc_profile = document_profile

                    st.success("Document processed successfully.")

            except Exception as exc:
                st.error(f"Error: {exc}")

    if st.session_state.doc_info:
        doc_info = st.session_state.doc_info
        st.markdown(
            f"**File:** {escape(str(doc_info['filename']))}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{doc_info['pages']} pages, {doc_info['chunks']} chunks"
        )
        st.caption(f"Uploaded {doc_info['uploaded_at']}")

        if st.session_state.doc_profile:
            with st.expander("Document Profile"):
                profile = st.session_state.doc_profile
                st.markdown(f"**Header:** {profile.get('title', 'Unknown')}")

                sections = profile.get("sections") or []
                if sections:
                    st.caption("Sections")
                    st.write(", ".join(sections[:12]))

                notes = profile.get("notes") or []
                if notes:
                    st.caption("Notes")
                    for note in notes[:4]:
                        st.markdown(f"- {escape(note)}")

                key_items = []
                for label, values in (
                    ("Emails", profile.get("emails") or []),
                    ("Phones", profile.get("phones") or []),
                    ("Links", profile.get("links") or []),
                ):
                    if values:
                        key_items.append(f"**{label}:** {', '.join(values)}")

                if key_items:
                    st.caption("Key Info")
                    for item in key_items:
                        st.markdown(item)

        if st.button("Clear Document", use_container_width=True):
            st.session_state.doc_info = None
            st.session_state.messages = []
            st.session_state.doc_chunks = []
            st.session_state.doc_profile = None
            st.rerun()
    else:
        st.caption("No document loaded.")

    st.divider()

    st.markdown("### Retrieval")

    top_k = st.slider(
        "Top K Chunks",
        1,
        10,
        5,
        disabled=not settings_ready,
    )

    model = st.selectbox(
        "Groq Model",
        GROQ_MODELS,
        index=GROQ_MODELS.index(DEFAULT_GROQ_MODEL),
        disabled=not settings_ready,
    )

    show_sources = st.toggle(
        "Show Sources",
        value=True,
    )

    st.caption(
        "Embeddings: "
        f"{app_settings.OPENROUTER_EMBEDDING_MODEL} "
        f"({app_settings.OPENROUTER_EMBEDDING_DIMENSIONS}d)"
    )

    st.divider()

    credentials = render_settings(settings_ready)
    apply_runtime_settings(credentials)
    missing_settings = missing_required_settings(credentials)
    settings_ready = not missing_settings and connection_error is None

header_col, action_col = st.columns([5, 1])

with header_col:
    st.markdown(
        """
<div class="app-kicker">Document retrieval workspace</div>
<h1 class="app-title">EAZI'S RAG Chatbot</h1>
<div class="app-subtitle">Search uploaded documents and answer from retrieved context.</div>
""",
        unsafe_allow_html=True,
    )

with action_col:
    st.write("")
    st.write("")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if settings_ready:
        render_status_card(
            "Credentials",
            "Ready",
            (
                "Groq primary, OpenRouter fallback, "
                f"{app_settings.OPENROUTER_EMBEDDING_MODEL}"
            ),
            "ready",
        )
    elif connection_error:
        render_status_card(
            "Credentials",
            "Connection issue",
            connection_error,
            "error",
        )
    else:
        render_status_card(
            "Credentials",
            "Incomplete",
            "Missing " + ", ".join(missing_settings),
            "warn",
        )

with status_col2:
    if st.session_state.doc_info:
        doc_info = st.session_state.doc_info
        render_status_card(
            "Document",
            doc_info["filename"],
            f"{doc_info['pages']} pages, {doc_info['chunks']} chunks",
            "ready",
        )
    else:
        render_status_card(
            "Document",
            "Not loaded",
            "Upload and process a file from the sidebar",
            "warn",
        )

with status_col3:
    render_status_card(
        "Retrieval",
        model,
        (
            f"Top {top_k} chunks, sources {'on' if show_sources else 'off'}, "
            f"fallback {app_settings.OPENROUTER_CHAT_MODEL}"
        ),
        "info",
    )

st.divider()

chat_title_col, chat_action_col = st.columns([5, 1])

with chat_title_col:
    st.markdown("### Chat")

with chat_action_col:
    st.caption(f"{len(st.session_state.messages)} messages")

if st.session_state.messages:
    for msg in st.session_state.messages:
        render_message(msg, show_sources)
else:
    if not settings_ready:
        empty_title = "Settings are required"
        empty_detail = "Open Settings in the sidebar and add the required credentials."
    elif not st.session_state.doc_info:
        empty_title = "No document loaded"
        empty_detail = "Upload and process a document from the sidebar."
    else:
        empty_title = "Ready for questions"
        empty_detail = "Ask about the uploaded document."

    st.markdown(
        f"""
<div class="chat-empty">
    <strong>{escape(empty_title)}</strong>
    {escape(empty_detail)}
</div>
""",
        unsafe_allow_html=True,
    )

prompt = st.chat_input(
    "Ask about the current document",
    disabled=not settings_ready or not st.session_state.doc_info,
)

if prompt:
    if supabase is None:
        st.warning("Complete settings before asking questions.")
    else:
        time_now = datetime.now().strftime("%I:%M %p")

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "time": time_now,
            }
        )

        with st.spinner("Searching document..."):
            try:
                answer_provider = None
                fallback_reason = None
                document_chunks = st.session_state.doc_chunks

                if not document_chunks:
                    document_chunks = fetch_document_chunks(
                        supabase,
                        st.session_state.doc_info,
                    )
                    st.session_state.doc_chunks = document_chunks

                if not st.session_state.doc_profile and document_chunks:
                    st.session_state.doc_profile = build_document_profile(
                        "\n".join(document_chunks),
                        st.session_state.doc_info["filename"],
                    )

                query_embedding = get_embedding([prompt])[0]

                if hasattr(query_embedding, "tolist"):
                    query_embedding = query_embedding.tolist()

                result = supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_count": max(top_k * 3, top_k + 5),
                    },
                ).execute()

                retrieved_rows = [
                    row
                    for row in result.data or []
                    if row_belongs_to_current_document(
                        row,
                        st.session_state.doc_info,
                    )
                ][:top_k]

                if retrieved_rows:
                    matched_chunks = [row["content"] for row in retrieved_rows]

                    sources = [
                        row.get("metadata", {}).get(
                            "filename",
                            "Unknown source",
                        )
                        for row in retrieved_rows
                    ]
                else:
                    matched_chunks = rank_fallback_chunks(
                        prompt,
                        document_chunks,
                        top_k,
                    )

                    if not matched_chunks and not document_chunks:
                        answer = "No relevant information found."
                        sources = []
                        context = ""
                        answer_provider = "Retrieval"
                    else:
                        context = build_answer_context(
                            st.session_state.doc_info,
                            document_chunks,
                            matched_chunks,
                            st.session_state.doc_profile,
                        )
                        sources = [st.session_state.doc_info["filename"]]

                if retrieved_rows:
                    context = build_answer_context(
                        st.session_state.doc_info,
                        document_chunks,
                        matched_chunks,
                        st.session_state.doc_profile,
                    )

                if context:
                    rag_answer = get_rag_answer(
                        question=prompt,
                        context=context,
                        groq_api_key=credentials["GROQ_API_KEY"],
                        openrouter_api_key=credentials["OPENROUTER_API_KEY"],
                        model=model,
                    )
                    answer = rag_answer.content
                    answer_provider = (
                        f"{rag_answer.provider}: {rag_answer.model}"
                    )
                    fallback_reason = rag_answer.fallback_reason

            except Exception as exc:
                answer = f"Error: {exc}"
                sources = []
                answer_provider = "Error"
                fallback_reason = None

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "provider": answer_provider,
                "fallback_reason": fallback_reason,
                "time": datetime.now().strftime("%I:%M %p"),
            }
        )

        st.rerun()

st.divider()

st.caption(
    "Built with Streamlit, LangChain, Supabase, Groq, OpenRouter, "
    "and Gemini Embedding 001"
)

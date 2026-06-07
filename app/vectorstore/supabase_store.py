from supabase import create_client

from app.config.settings import get_setting
from embeddings.embedding_model import get_embedding


def get_db():
    supabase_url = get_setting("SUPABASE_URL")
    supabase_key = get_setting("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY.")

    return create_client(supabase_url, supabase_key)


def retrieve_chunks(question, top_k=5):
    question_embedding = get_embedding(question)[0]

    response = get_db().rpc(
        "match_documents",
        {
            "query_embedding": question_embedding,
            "match_count": top_k,
        },
    ).execute()

    return response.data


def store_embeddings(chunks, embeddings, filename=None):
    data = []

    for chunk, embedding in zip(chunks, embeddings):
        data.append(
            {
                "content": chunk,
                "embedding": embedding,
                "metadata": {
                    "filename": filename,
                },
            }
        )

    get_db().table("documents").insert(data).execute()

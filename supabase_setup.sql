create extension if not exists vector with schema extensions;

create table if not exists public.documents (
    id bigserial primary key,
    content text not null,
    embedding vector(1536) not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists documents_embedding_idx
on public.documents
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

create or replace function public.match_documents(
    query_embedding vector(1536),
    match_count int default 5
)
returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    set local ivfflat.probes = 100;

    return query
    select
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from public.documents
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;

alter table public.documents enable row level security;

create policy "Allow document reads"
on public.documents
for select
to anon
using (true);

create policy "Allow document inserts"
on public.documents
for insert
to anon
with check (true);

grant usage on schema public to anon;
grant select, insert on public.documents to anon;
grant usage, select on sequence public.documents_id_seq to anon;
grant execute on function public.match_documents(vector, int) to anon;

notify pgrst, 'reload schema';

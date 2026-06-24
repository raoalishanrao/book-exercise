-- =============================================================================
-- Educational Textbook RAG Schema (Supabase + pgvector)
-- Supports class 9, 10, 11 books with Gemini embeddings (768-dim)
-- Models: gemini-2.5-flash (bot), text-embedding-004 (embeddings)
-- =============================================================================

-- Extensions
create extension if not exists "uuid-ossp";
create extension if not exists vector;
create extension if not exists pg_trgm;

-- =============================================================================
-- ENUMS
-- =============================================================================

create type chunk_type as enum (
  'theory',              -- explanatory text
  'definition',          -- key definitions
  'formula',             -- equations / laws
  'example',             -- worked examples
  'problem_statement',   -- exercise question only
  'solution',            -- full solution
  'hint',                -- progressive tutoring hints
  'summary',             -- chapter/section summary
  'figure_caption'       -- diagram descriptions
);

create type content_unit_type as enum (
  'section',
  'example',
  'exercise',
  'review_question',
  'activity'
);

create type ingestion_status as enum (
  'pending',
  'extracting',
  'chunking',
  'embedding',
  'injecting',
  'completed',
  'failed'
);

create type message_role as enum (
  'student',
  'assistant',
  'system'
);

-- =============================================================================
-- REFERENCE / CATALOG TABLES (extensible for 9, 10, 11)
-- =============================================================================

create table academic_classes (
  id          uuid primary key default uuid_generate_v4(),
  grade       smallint not null unique check (grade between 1 and 12),
  label       text not null,                    -- e.g. 'Class 9', 'Class 10'
  created_at  timestamptz not null default now()
);

create table subjects (
  id          uuid primary key default uuid_generate_v4(),
  code        text not null unique,             -- e.g. 'physics', 'chemistry'
  name        text not null,                    -- e.g. 'Physics'
  created_at  timestamptz not null default now()
);

create table books (
  id              uuid primary key default uuid_generate_v4(),
  class_id        uuid not null references academic_classes(id) on delete restrict,
  subject_id      uuid not null references subjects(id) on delete restrict,
  title           text not null,
  edition         text,
  publisher       text,
  language        text not null default 'en',
  isbn            text,
  source_file     text,                         -- e.g. 'Physics 9.pdf'
  total_pages     integer,
  metadata        jsonb not null default '{}',
  is_active       boolean not null default true,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  unique (class_id, subject_id, title)
);

create table chapters (
  id              uuid primary key default uuid_generate_v4(),
  book_id         uuid not null references books(id) on delete cascade,
  chapter_number  smallint not null,
  title           text not null,
  page_start      integer,
  page_end        integer,
  sort_order      integer not null default 0,
  metadata        jsonb not null default '{}',
  created_at      timestamptz not null default now(),
  unique (book_id, chapter_number)
);

create table sections (
  id              uuid primary key default uuid_generate_v4(),
  chapter_id      uuid not null references chapters(id) on delete cascade,
  section_number  text,                         -- e.g. '3.2'
  title           text,
  page_start      integer,
  page_end        integer,
  sort_order      integer not null default 0,
  metadata        jsonb not null default '{}',
  created_at      timestamptz not null default now()
);

-- Logical content units (exercises, examples) — parent for related chunks
create table content_units (
  id              uuid primary key default uuid_generate_v4(),
  book_id         uuid not null references books(id) on delete cascade,
  chapter_id      uuid references chapters(id) on delete set null,
  section_id      uuid references sections(id) on delete set null,
  unit_type       content_unit_type not null,
  unit_ref        text,                         -- e.g. 'Exercise 4.3', 'Example 2.1'
  title           text,
  page_number     integer,
  difficulty      smallint check (difficulty between 1 and 5),
  topics          text[] not null default '{}', -- e.g. {'kinematics','velocity'}
  metadata        jsonb not null default '{}',
  created_at      timestamptz not null default now()
);

-- Links between units (problem → prerequisite theory, similar problems)
create table content_unit_relations (
  id                  uuid primary key default uuid_generate_v4(),
  source_unit_id      uuid not null references content_units(id) on delete cascade,
  target_unit_id      uuid not null references content_units(id) on delete cascade,
  relation_type       text not null check (relation_type in (
    'prerequisite', 'related_theory', 'similar_problem', 'follow_up'
  )),
  created_at          timestamptz not null default now(),
  unique (source_unit_id, target_unit_id, relation_type)
);

-- =============================================================================
-- RAG CHUNKS + EMBEDDINGS
-- =============================================================================

create table document_chunks (
  id                  uuid primary key default uuid_generate_v4(),
  -- Denormalized filters for fast scoped retrieval (class/book/subject)
  class_id            uuid not null references academic_classes(id) on delete restrict,
  subject_id          uuid not null references subjects(id) on delete restrict,
  book_id             uuid not null references books(id) on delete cascade,
  chapter_id          uuid references chapters(id) on delete set null,
  section_id          uuid references sections(id) on delete set null,
  content_unit_id     uuid references content_units(id) on delete set null,

  chunk_type          chunk_type not null,
  chunk_index         integer not null default 0,   -- order within parent unit/section
  content             text not null,
  content_hash        text not null,                -- sha256 for dedup on re-ingest
  token_count         integer,
  page_start          integer,
  page_end            integer,

  -- Structured fields for problem-solving retrieval
  problem_number      text,                         -- e.g. '4.3'
  has_solution        boolean not null default false,
  topics              text[] not null default '{}',

  metadata            jsonb not null default '{}',
  -- Gemini text-embedding-004 default dimension
  embedding           vector(768),
  embedding_model     text not null default 'text-embedding-004',

  -- Full-text search (formulas, exact terms, problem numbers)
  content_tsv         tsvector generated always as (
    setweight(to_tsvector('english', coalesce(problem_number, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(content, '')), 'B')
  ) stored,

  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now(),

  unique (book_id, content_hash)
);

-- Chunk-to-chunk links (problem_statement → solution, theory cross-refs)
create table chunk_relations (
  id              uuid primary key default uuid_generate_v4(),
  source_chunk_id uuid not null references document_chunks(id) on delete cascade,
  target_chunk_id uuid not null references document_chunks(id) on delete cascade,
  relation_type   text not null check (relation_type in (
    'solution_for', 'hint_for', 'theory_for', 'example_for', 'continues', 'references'
  )),
  created_at      timestamptz not null default now(),
  unique (source_chunk_id, target_chunk_id, relation_type)
);

-- =============================================================================
-- INGESTION PIPELINE TRACKING
-- =============================================================================

create table ingestion_jobs (
  id              uuid primary key default uuid_generate_v4(),
  book_id         uuid not null references books(id) on delete cascade,
  status          ingestion_status not null default 'pending',
  source_file     text not null,
  total_chunks    integer not null default 0,
  embedded_chunks integer not null default 0,
  error_message   text,
  started_at      timestamptz,
  completed_at    timestamptz,
  metadata        jsonb not null default '{}',
  created_at      timestamptz not null default now()
);

create table ingestion_job_logs (
  id          uuid primary key default uuid_generate_v4(),
  job_id      uuid not null references ingestion_jobs(id) on delete cascade,
  stage       ingestion_status not null,
  message     text not null,
  details     jsonb not null default '{}',
  created_at  timestamptz not null default now()
);

-- =============================================================================
-- TUTORING BOT — CONVERSATIONS (scoped to class + book)
-- =============================================================================

create table students (
  id          uuid primary key default uuid_generate_v4(),
  external_id text unique,                      -- your auth user id
  display_name text,
  created_at  timestamptz not null default now()
);

create table conversation_sessions (
  id              uuid primary key default uuid_generate_v4(),
  student_id      uuid references students(id) on delete set null,
  class_id        uuid not null references academic_classes(id) on delete restrict,
  book_id         uuid not null references books(id) on delete restrict,
  subject_id      uuid not null references subjects(id) on delete restrict,
  -- Optional: student is working on a specific problem
  content_unit_id uuid references content_units(id) on delete set null,
  title           text,
  metadata        jsonb not null default '{}',
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create table conversation_messages (
  id                  uuid primary key default uuid_generate_v4(),
  session_id          uuid not null references conversation_sessions(id) on delete cascade,
  role                message_role not null,
  content             text not null,
  -- Audit trail: which chunks grounded this reply
  retrieved_chunk_ids uuid[] not null default '{}',
  model               text not null default 'gemini-2.5-flash',
  prompt_tokens       integer,
  completion_tokens   integer,
  metadata            jsonb not null default '{}',
  created_at          timestamptz not null default now()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

create index idx_books_class_subject on books (class_id, subject_id) where is_active;
create index idx_chapters_book on chapters (book_id, sort_order);
create index idx_sections_chapter on sections (chapter_id, sort_order);
create index idx_content_units_book on content_units (book_id, unit_type);
create index idx_content_units_topics on content_units using gin (topics);

create index idx_chunks_book_type on document_chunks (book_id, chunk_type);
create index idx_chunks_class_book on document_chunks (class_id, book_id);
create index idx_chunks_content_unit on document_chunks (content_unit_id) where content_unit_id is not null;
create index idx_chunks_topics on document_chunks using gin (topics);
create index idx_chunks_tsv on document_chunks using gin (content_tsv);
create index idx_chunks_problem on document_chunks (book_id, problem_number) where problem_number is not null;

-- HNSW index for cosine similarity (best for normalized embeddings)
create index idx_chunks_embedding_hnsw
  on document_chunks
  using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create index idx_ingestion_jobs_book_status on ingestion_jobs (book_id, status);
create index idx_sessions_student on conversation_sessions (student_id, created_at desc);
create index idx_messages_session on conversation_messages (session_id, created_at);

-- =============================================================================
-- ROW LEVEL SECURITY (enable in Supabase dashboard; policies are examples)
-- =============================================================================

alter table academic_classes enable row level security;
alter table subjects enable row level security;
alter table books enable row level security;
alter table chapters enable row level security;
alter table sections enable row level security;
alter table content_units enable row level security;
alter table document_chunks enable row level security;
alter table conversation_sessions enable row level security;
alter table conversation_messages enable row level security;

-- Service role bypasses RLS. Add authenticated policies as needed, e.g.:
-- create policy "Students read own sessions"
--   on conversation_sessions for select
--   using (student_id = auth.uid());

-- =============================================================================
-- SEED DATA: Class 9–11 + Physics
-- =============================================================================

insert into academic_classes (grade, label) values
  (9,  'Class 9'),
  (10, 'Class 10'),
  (11, 'Class 11')
on conflict (grade) do nothing;

insert into subjects (code, name) values
  ('physics',   'Physics'),
  ('chemistry', 'Chemistry'),
  ('mathematics', 'Mathematics'),
  ('biology',   'Biology')
on conflict (code) do nothing;

-- Register Class 9 Physics book (update source_file path as needed)
insert into books (class_id, subject_id, title, source_file, publisher)
select
  c.id,
  s.id,
  'Physics Class 9',
  'Physics 9.pdf',
  null
from academic_classes c
cross join subjects s
where c.grade = 9 and s.code = 'physics'
on conflict (class_id, subject_id, title) do nothing;

-- =============================================================================
-- RPC: Hybrid semantic search (vector + optional full-text boost)
-- Scoped by class/book for accurate, curriculum-specific answers
-- =============================================================================

create or replace function match_document_chunks (
  query_embedding     vector(768),
  match_count         integer default 12,
  filter_class_id     uuid default null,
  filter_book_id      uuid default null,
  filter_subject_id   uuid default null,
  filter_chapter_id   uuid default null,
  filter_chunk_types  chunk_type[] default null,
  filter_topics       text[] default null,
  similarity_threshold float default 0.55,
  text_query          text default null          -- optional keyword boost
)
returns table (
  id                  uuid,
  book_id             uuid,
  chapter_id          uuid,
  content_unit_id     uuid,
  chunk_type          chunk_type,
  content             text,
  problem_number      text,
  topics              text[],
  metadata            jsonb,
  similarity          float,
  text_rank           float
)
language plpgsql
stable
as $$
begin
  return query
  select
    dc.id,
    dc.book_id,
    dc.chapter_id,
    dc.content_unit_id,
    dc.chunk_type,
    dc.content,
    dc.problem_number,
    dc.topics,
    dc.metadata,
    (1 - (dc.embedding <=> query_embedding))::float as similarity,
    case
      when text_query is not null and text_query <> ''
      then ts_rank(dc.content_tsv, plainto_tsquery('english', text_query))::float
      else 0::float
    end as text_rank
  from document_chunks dc
  where dc.embedding is not null
    and (filter_class_id is null or dc.class_id = filter_class_id)
    and (filter_book_id is null or dc.book_id = filter_book_id)
    and (filter_subject_id is null or dc.subject_id = filter_subject_id)
    and (filter_chapter_id is null or dc.chapter_id = filter_chapter_id)
    and (filter_chunk_types is null or dc.chunk_type = any(filter_chunk_types))
    and (filter_topics is null or dc.topics && filter_topics)
    and (1 - (dc.embedding <=> query_embedding)) >= similarity_threshold
  order by
  -- Blend semantic + keyword when text_query provided
    case
      when text_query is not null and text_query <> ''
      then (1 - (dc.embedding <=> query_embedding)) * 0.75
         + ts_rank(dc.content_tsv, plainto_tsquery('english', text_query)) * 0.25
      else (1 - (dc.embedding <=> query_embedding))
    end desc
  limit match_count;
end;
$$;

-- =============================================================================
-- RPC: Fetch problem context bundle for tutoring bot
-- Returns problem statement + linked theory + hints (not full solution first)
-- =============================================================================

create or replace function get_problem_context (
  p_content_unit_id uuid,
  include_solution  boolean default false
)
returns table (
  chunk_id    uuid,
  chunk_type  chunk_type,
  content     text,
  sort_order  integer
)
language sql
stable
as $$
  -- Problem statement first
  select dc.id as chunk_id, dc.chunk_type, dc.content, 1 as sort_order
  from document_chunks dc
  where dc.content_unit_id = p_content_unit_id
    and dc.chunk_type = 'problem_statement'

  union all

  -- Linked theory via chunk_relations
  select dc.id as chunk_id, dc.chunk_type, dc.content, 2 as sort_order
  from chunk_relations cr
  join document_chunks dc on dc.id = cr.target_chunk_id
  where cr.source_chunk_id in (
    select id from document_chunks
    where content_unit_id = p_content_unit_id
      and chunk_type = 'problem_statement'
  )
  and cr.relation_type = 'theory_for'

  union all

  -- Hints in order
  select dc.id as chunk_id, dc.chunk_type, dc.content, (3 + dc.chunk_index) as sort_order
  from document_chunks dc
  where dc.content_unit_id = p_content_unit_id
    and dc.chunk_type = 'hint'

  union all

  -- Solution only when explicitly requested
  select dc.id as chunk_id, dc.chunk_type, dc.content, 99 as sort_order
  from document_chunks dc
  where dc.content_unit_id = p_content_unit_id
    and dc.chunk_type = 'solution'
    and include_solution = true

  order by sort_order;
$$;

-- =============================================================================
-- RPC: Related chunks for a retrieved chunk (expand context window)
-- =============================================================================

create or replace function get_related_chunks (
  p_chunk_id uuid,
  p_relation_types text[] default array['theory_for', 'solution_for', 'example_for', 'continues']
)
returns table (
  chunk_id    uuid,
  chunk_type  chunk_type,
  content     text,
  relation_type text
)
language sql
stable
as $$
  select
    dc.id,
    dc.chunk_type,
    dc.content,
    cr.relation_type
  from chunk_relations cr
  join document_chunks dc on dc.id = cr.target_chunk_id
  where cr.source_chunk_id = p_chunk_id
    and cr.relation_type = any(p_relation_types);
$$;

-- =============================================================================
-- TRIGGERS
-- =============================================================================

create or replace function set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger trg_books_updated_at
  before update on books
  for each row execute function set_updated_at();

create trigger trg_chunks_updated_at
  before update on document_chunks
  for each row execute function set_updated_at();

create trigger trg_sessions_updated_at
  before update on conversation_sessions
  for each row execute function set_updated_at();

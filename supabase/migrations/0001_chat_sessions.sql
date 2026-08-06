-- Chat history: per-user sessions and messages.
--
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
-- Until it is run, the dashboard falls back to browser-local storage, so nothing
-- breaks — history simply does not follow the user to another device.
--
-- Messages are rows rather than a JSONB blob on the session. Appending a reply is
-- then a single INSERT, so two open tabs cannot overwrite each other's turns the
-- way a read-modify-write of one JSON document would.

create table if not exists public.chat_sessions (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users (id) on delete cascade,
    title       text not null default 'New session',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table if not exists public.chat_messages (
    id          bigint generated always as identity primary key,
    session_id  uuid not null references public.chat_sessions (id) on delete cascade,
    -- Denormalised so a row can be authorised without joining to the session.
    user_id     uuid not null references auth.users (id) on delete cascade,
    role        text not null check (role in ('user', 'assistant')),
    content     text not null,
    route       text,
    grounded    boolean,
    sources     jsonb,
    is_error    boolean not null default false,
    created_at  timestamptz not null default now()
);

-- Sidebar lists sessions newest-first; a session view reads messages in order.
create index if not exists chat_sessions_user_updated_idx
    on public.chat_sessions (user_id, updated_at desc);
create index if not exists chat_messages_session_created_idx
    on public.chat_messages (session_id, created_at);

-- ── Row-Level Security ──────────────────────────────────────────────────────
-- The browser talks to these tables directly with the user's own access token,
-- so RLS is the only thing standing between tenants. There is no service-role
-- key in this application to bypass it.
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;

drop policy if exists "own sessions" on public.chat_sessions;
create policy "own sessions" on public.chat_sessions
    for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "own messages" on public.chat_messages;
create policy "own messages" on public.chat_messages
    for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- Keep updated_at honest so the sidebar's ordering and "2h ago" labels are real
-- rather than dependent on the client remembering to set them.
create or replace function public.touch_chat_session()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.chat_sessions
       set updated_at = now()
     where id = new.session_id;
    return new;
end;
$$;

drop trigger if exists chat_messages_touch_session on public.chat_messages;
create trigger chat_messages_touch_session
    after insert on public.chat_messages
    for each row execute function public.touch_chat_session();

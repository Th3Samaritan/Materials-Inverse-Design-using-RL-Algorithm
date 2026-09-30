-- Additive migration: safe to share the Mineral Atlas Supabase project.
-- Does not change mineral tables, buckets, users, or Auth configuration.
begin;
create table public.mast_runs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    created_at timestamptz not null default now(),
    report jsonb not null,
    constraint mast_report_object check (jsonb_typeof(report) = 'object'),
    constraint mast_report_size check (octet_length(report::text) <= 131072)
);
create index mast_runs_owner_created on public.mast_runs(user_id, created_at desc);
alter table public.mast_runs enable row level security;
revoke all on public.mast_runs from anon, authenticated;
grant select, insert on public.mast_runs to authenticated;
create policy mast_runs_read_own on public.mast_runs for select to authenticated
    using ((select auth.uid()) = user_id);
create policy mast_runs_insert_own on public.mast_runs for insert to authenticated
    with check ((select auth.uid()) = user_id);
comment on table public.mast_runs is 'Owner-scoped MAST research reports; user-owned records, not certified laboratory results.';
commit;

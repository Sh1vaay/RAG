-- =============================================================================
-- Phase 3: Supabase Storage — bucket + RLS policies
-- Run this once in the Supabase SQL Editor (Dashboard → SQL Editor → New Query).
-- =============================================================================

-- 1. Create a private bucket called "workspaces".
--    Private means every download requires a valid JWT; no anonymous access.
INSERT INTO storage.buckets (id, name, public, file_size_limit)
VALUES ('workspaces', 'workspaces', false, 52428800)   -- 50 MB limit
ON CONFLICT (id) DO NOTHING;

-- 2. RLS policies on storage.objects.
--    Path convention:  {user_id}/documents/{file}
--                      {user_id}/faiss_db/{file}
--                      {user_id}/config.json
--
--    (storage.foldername(name))[1] extracts the first path segment, which is
--    always the user id.  Comparing it to auth.uid()::text ensures each user
--    can only touch their own folder.

-- SELECT — download / list
CREATE POLICY "Users read own files"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (
    bucket_id = 'workspaces'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

-- INSERT — upload new files
CREATE POLICY "Users upload own files"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (
    bucket_id = 'workspaces'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

-- UPDATE — overwrite (upsert) existing files
CREATE POLICY "Users update own files"
  ON storage.objects FOR UPDATE
  TO authenticated
  USING (
    bucket_id = 'workspaces'
    AND (storage.foldername(name))[1] = auth.uid()::text
  )
  WITH CHECK (
    bucket_id = 'workspaces'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

-- DELETE — remove files
CREATE POLICY "Users delete own files"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (
    bucket_id = 'workspaces'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

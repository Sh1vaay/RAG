"use client";

import { createClient, type SupabaseClient } from "@supabase/supabase-js";

/**
 * Browser Supabase client.
 *
 * The publishable key is meant to ship in the bundle — it is not a secret. What
 * protects a user's data is their session token plus row-level security, not the
 * secrecy of this key. There is no service-role key anywhere in this app.
 */
const url = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const publishableKey =
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??
  "";

/** When unset, the app runs single-user with no sign-in — matching the backend. */
export const authConfigured = Boolean(url && publishableKey);

let client: SupabaseClient | null = null;

export function supabase(): SupabaseClient {
  if (!authConfigured) {
    throw new Error(
      "Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and " +
        "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY in frontend/.env.local.",
    );
  }
  if (!client) {
    client = createClient(url, publishableKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    });
  }
  return client;
}

/** Current access token, or null when signed out / not configured. */
export async function accessToken(): Promise<string | null> {
  if (!authConfigured) return null;
  const { data } = await supabase().auth.getSession();
  return data.session?.access_token ?? null;
}

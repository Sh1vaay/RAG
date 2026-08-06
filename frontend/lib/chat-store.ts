"use client";

/**
 * Chat history persistence.
 *
 * Local-first: the browser copy is written synchronously so the UI never waits on
 * the network, and Supabase is updated in the background. On sign-in the remote
 * copy wins, which is what makes history follow a user to another device.
 *
 * Every remote call degrades to local-only on failure — no Supabase configured,
 * the migration not yet run, or simply offline. `remoteAvailable()` reports which
 * mode is in effect so the UI can be honest about it.
 */

import { authConfigured, supabase } from "./supabase";
import type { ChatMessage, Session } from "./sessions";
import { newSession } from "./sessions";

let remoteOk: boolean | null = null;

/** null until the first remote call has told us whether the tables exist. */
export function remoteAvailable(): boolean | null {
  return remoteOk;
}

/** A missing table means the migration has not been run — expected, not an error. */
function noteFailure(err: unknown, action: string) {
  const message = (err as { message?: string })?.message ?? String(err);
  const missingTable = /relation .* does not exist|schema cache|PGRST205/i.test(message);
  if (remoteOk !== false) {
    console.warn(
      missingTable
        ? `[chat] Supabase chat tables not found — run supabase/migrations/0001_chat_sessions.sql. ` +
            `Falling back to browser-local history.`
        : `[chat] ${action} failed, using local history only: ${message}`,
    );
  }
  remoteOk = false;
}

interface MessageRow {
  role: string;
  content: string;
  route: string | null;
  grounded: boolean | null;
  sources: unknown;
  is_error: boolean;
}

function toMessage(row: MessageRow): ChatMessage {
  return {
    role: row.role === "user" ? "user" : "assistant",
    content: row.content,
    ...(row.route ? { route: row.route } : {}),
    ...(row.grounded === null ? {} : { grounded: row.grounded }),
    ...(Array.isArray(row.sources) ? { sources: row.sources as ChatMessage["sources"] } : {}),
    ...(row.is_error ? { error: true } : {}),
  };
}

/** Pull every session for the signed-in user, newest first. */
export async function fetchRemoteSessions(): Promise<Session[] | null> {
  if (!authConfigured) return null;
  try {
    const client = supabase();
    const { data: rows, error } = await client
      .from("chat_sessions")
      .select("id, title, updated_at, chat_messages(role, content, route, grounded, sources, is_error, created_at)")
      .order("updated_at", { ascending: false })
      .order("created_at", { referencedTable: "chat_messages", ascending: true });

    if (error) throw error;
    remoteOk = true;

    return (rows ?? []).map((row) => ({
      id: row.id as string,
      title: (row.title as string) || "New session",
      updatedAt: new Date(row.updated_at as string).getTime(),
      messages: ((row.chat_messages ?? []) as MessageRow[]).map(toMessage),
    }));
  } catch (err) {
    noteFailure(err, "Loading history");
    return null;
  }
}

/** Create the session row. Safe to call repeatedly — upsert keyed on id. */
export async function pushSession(session: Session): Promise<void> {
  if (!authConfigured || remoteOk === false) return;
  try {
    const client = supabase();
    const { data: auth } = await client.auth.getUser();
    const userId = auth.user?.id;
    if (!userId) return;

    const { error } = await client
      .from("chat_sessions")
      .upsert({ id: session.id, user_id: userId, title: session.title }, { onConflict: "id" });
    if (error) throw error;
    remoteOk = true;
  } catch (err) {
    noteFailure(err, "Saving session");
  }
}

/** Append one turn. An INSERT, so concurrent tabs cannot overwrite each other. */
export async function pushMessage(sessionId: string, message: ChatMessage): Promise<void> {
  if (!authConfigured || remoteOk === false) return;
  try {
    const client = supabase();
    const { data: auth } = await client.auth.getUser();
    const userId = auth.user?.id;
    if (!userId) return;

    const { error } = await client.from("chat_messages").insert({
      session_id: sessionId,
      user_id: userId,
      role: message.role,
      content: message.content,
      route: message.route ?? null,
      grounded: message.grounded ?? null,
      sources: message.sources ?? null,
      is_error: Boolean(message.error),
    });
    if (error) throw error;
    remoteOk = true;
  } catch (err) {
    noteFailure(err, "Saving message");
  }
}

export async function renameRemoteSession(sessionId: string, title: string): Promise<void> {
  if (!authConfigured || remoteOk === false) return;
  try {
    const { error } = await supabase()
      .from("chat_sessions")
      .update({ title })
      .eq("id", sessionId);
    if (error) throw error;
  } catch (err) {
    noteFailure(err, "Renaming session");
  }
}

/** Messages cascade via the foreign key, so one delete is enough. */
export async function deleteRemoteSession(sessionId: string): Promise<void> {
  if (!authConfigured || remoteOk === false) return;
  try {
    const { error } = await supabase().from("chat_sessions").delete().eq("id", sessionId);
    if (error) throw error;
  } catch (err) {
    noteFailure(err, "Deleting session");
  }
}

export async function deleteAllRemoteSessions(): Promise<void> {
  if (!authConfigured || remoteOk === false) return;
  try {
    const client = supabase();
    const { data: auth } = await client.auth.getUser();
    if (!auth.user?.id) return;
    const { error } = await client.from("chat_sessions").delete().eq("user_id", auth.user.id);
    if (error) throw error;
  } catch (err) {
    noteFailure(err, "Clearing history");
  }
}

/**
 * One-time upload of sessions created before the migration, so existing history
 * is not stranded in the browser. Only non-empty sessions are worth moving.
 */
export async function migrateLocalSessions(local: Session[]): Promise<void> {
  if (!authConfigured || remoteOk === false) return;
  for (const session of local.filter((s) => s.messages.length > 0)) {
    await pushSession(session);
    for (const message of session.messages) {
      await pushMessage(session.id, message);
    }
  }
}

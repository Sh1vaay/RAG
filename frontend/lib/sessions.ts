/**
 * Chat sessions, persisted to localStorage.
 *
 * The backend is stateless — it takes the full history on every call — so
 * session storage lives entirely on the client until there is a user store.
 */
import type { SourceDocument } from "./api";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  route?: string;
  sources?: SourceDocument[];
  error?: boolean;
  /** False when the answer is general knowledge, not from the user's documents. */
  grounded?: boolean;
}

export interface Session {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
}

/**
 * Storage key, scoped to the signed-in user.
 *
 * This used to be one global key with no user in it, and sign-out did not clear
 * it — so on a shared machine the next person to sign in saw the previous user's
 * entire history, including document excerpts in `sources`. Scoping the key and
 * clearing on sign-out closes that.
 */
const LEGACY_KEY = "aether.sessions.v2";
const storeKey = (userId: string) => `aether.sessions.v3.${userId || "local"}`;

export function newSession(): Session {
  return {
    id:
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`,
    title: "New session",
    updatedAt: Date.now(),
    messages: [],
  };
}

function blank() {
  const s = newSession();
  return { sessions: [s], activeId: s.id };
}

export function loadSessions(userId: string): { sessions: Session[]; activeId: string } {
  if (typeof window === "undefined") return blank();
  try {
    const raw = JSON.parse(localStorage.getItem(storeKey(userId)) || "{}");
    const sessions: Session[] =
      Array.isArray(raw.sessions) && raw.sessions.length ? raw.sessions : [newSession()];
    const activeId = sessions.some((s) => s.id === raw.activeId) ? raw.activeId : sessions[0].id;
    return { sessions, activeId };
  } catch {
    return blank();
  }
}

export function saveSessions(userId: string, sessions: Session[], activeId: string) {
  try {
    localStorage.setItem(storeKey(userId), JSON.stringify({ sessions, activeId }));
  } catch (err) {
    console.warn("Could not persist sessions:", err);
  }
}

/** Wipe this browser's copy on sign-out so the next user starts clean. */
export function clearLocalSessions(userId?: string) {
  try {
    if (userId) localStorage.removeItem(storeKey(userId));
    // The old unscoped key predates per-user storage; drop it wherever it is found.
    localStorage.removeItem(LEGACY_KEY);
  } catch {
    /* storage unavailable — nothing to clear */
  }
}

/** Derive a session title from the first thing the user actually asked. */
export function titleFrom(text: string) {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > 44 ? `${clean.slice(0, 44)}…` : clean || "New session";
}

export function relativeTime(ts: number) {
  const mins = Math.floor((Date.now() - ts) / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

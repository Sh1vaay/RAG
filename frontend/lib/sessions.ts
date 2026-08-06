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
}

export interface Session {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
}

const STORE_KEY = "aether.sessions.v2";

export function newSession(): Session {
  return {
    id: `S${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
    title: "New session",
    updatedAt: Date.now(),
    messages: [],
  };
}

export function loadSessions(): { sessions: Session[]; activeId: string } {
  if (typeof window === "undefined") {
    const s = newSession();
    return { sessions: [s], activeId: s.id };
  }
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    const sessions: Session[] = Array.isArray(raw.sessions) && raw.sessions.length
      ? raw.sessions
      : [newSession()];
    const activeId = sessions.some((s) => s.id === raw.activeId)
      ? raw.activeId
      : sessions[0].id;
    return { sessions, activeId };
  } catch {
    const s = newSession();
    return { sessions: [s], activeId: s.id };
  }
}

export function saveSessions(sessions: Session[], activeId: string) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ sessions, activeId }));
  } catch (err) {
    console.warn("Could not persist sessions:", err);
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

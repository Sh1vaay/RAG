/**
 * Typed client for the FastAPI backend.
 *
 * In production the Next export is served by FastAPI itself, so same-origin
 * works. In `next dev` the app runs on :3000 while the API is on :8000, hence
 * the env override — the backend already allows all CORS origins.
 */
import { accessToken } from "./supabase";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8000"
    : "");

export interface StagedFile {
  name: string;
  size: string;
  status: string;
}

export interface StatusResponse {
  status: string;
  database_loaded: boolean;
  document_chunks: number;
  routing_method: string;
  reranker_provider: string;
  langsmith_tracing: boolean;
  staged_files: StagedFile[];
  workspace?: string;
  user_email?: string | null;
  auth_enabled?: boolean;
  storage_enabled?: boolean;
  llm_provider?: string;
  llm_model?: string;
  llm_key_configured?: boolean;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_key_configured?: boolean;
  error?: string;
}

export interface ProviderEntry {
  id: string;
  default_model: string;
  key_env: string | null;
  requires_key: boolean;
  key_configured: boolean;
}

export interface ProviderCatalog {
  chat: ProviderEntry[];
  embedding: ProviderEntry[];
  chat_only: string[];
}

export interface SourceDocument {
  title: string;
  source: string;
  page: number | null;
  snippet: string;
}

export interface BuildMeta {
  id: string;
  status: "running" | "completed" | "failed";
  started_at: string;
  completed_at: string | null;
  raptor: boolean;
  error: string | null;
}

export interface ChatResponse {
  answer: string;
  route: string;
  sources: SourceDocument[];
  /** False when the answer came from the model's general knowledge because
   *  retrieval found nothing relevant in the user's documents. */
  grounded?: boolean;
}

export interface ConfigPayload {
  routing_method: string;
  reranker_provider: string;
  llm_provider?: string;
  llm_model?: string;
  embedding_provider?: string;
  embedding_model?: string;
  openai_key?: string | null;
  anthropic_key?: string | null;
  google_key?: string | null;
  xai_key?: string | null;
  cohere_key?: string | null;
}

/** FastAPI returns errors as `{ detail: string }`; surface that text, not a status code. */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** 401 means "sign in again", not "something broke" — callers redirect on this. */
  get isUnauthorized() {
    return this.status === 401;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  // Every call carries the session token. The backend derives the workspace from
  // its verified `sub` claim, so identity is never something we send in a body.
  const token = await accessToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${API_BASE || window.location.origin}. Is the server running?`,
      0,
    );
  }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw new ApiError(data?.detail ?? `Request failed (${res.status})`, res.status);
  }
  return data as T;
}

/** Public: tells the client whether a sign-in screen is needed at all. */
export interface AuthConfig {
  auth_enabled: boolean;
  supabase_url: string | null;
}

export async function fetchAuthConfig(): Promise<AuthConfig> {
  const res = await fetch(`${API_BASE}/api/auth/config`);
  if (!res.ok) return { auth_enabled: false, supabase_url: null };
  return res.json();
}

export const api = {
  status: () => request<StatusResponse>("/api/status"),

  providers: () => request<ProviderCatalog>("/api/providers"),

  chat: (message: string, history: { role: string; content: string }[]) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    }),

  config: (payload: ConfigPayload) =>
    request<{ status: string; message: string; embedding_changed: boolean }>("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  upload: (files: FileList | File[]) => {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    return request<{ status: string; uploaded_files: string[] }>("/api/upload", {
      method: "POST",
      body: form,
    });
  },

  ingest: (raptor: boolean) =>
    request<{ status: string; build_id: string }>(
      `/api/ingest?raptor=${raptor}`,
      { method: "POST" },
    ),

  listBuilds: () => request<{ builds: BuildMeta[] }>("/api/builds"),
  
  cancelBuild: (build_id: string) => 
    request<{ status: string }>(`/api/builds/${build_id}/cancel`, { method: "POST" }),

  deleteDocument: (filename: string) =>
    request<{ status: string; deleted: string }>(
      `/api/documents/${encodeURIComponent(filename)}`,
      { method: "DELETE" },
    ),
};

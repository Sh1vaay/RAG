"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  FolderOpen, LogOut, MessageSquare, Moon, Network, Plus, Settings2, Sun, Trash2, X,
} from "lucide-react";
import { useTheme } from "next-themes";
import { ApiError, api, fetchAuthConfig, type StatusResponse } from "@/lib/api";
import { authConfigured, supabase } from "@/lib/supabase";
import {
  loadSessions, newSession, relativeTime, saveSessions, type Session,
} from "@/lib/sessions";
import { ChatView } from "@/components/views/chat-view";
import { DocumentsView } from "@/components/views/documents-view";
import { SettingsView } from "@/components/views/settings-view";
import { StatusView } from "@/components/views/status-view";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type Tab = "chat" | "docs" | "status" | "settings";

const TABS: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "docs", label: "Documents", icon: FolderOpen },
  { id: "status", label: "Status", icon: Network },
  { id: "settings", label: "Settings", icon: Settings2 },
];

export default function Page() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("chat");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [mounted, setMounted] = useState(false);
  const [authReady, setAuthReady] = useState(false);
  const { resolvedTheme, setTheme } = useTheme();

  // Read initial tab from URL if present
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const urlTab = params.get("tab") as Tab;
      if (urlTab && ["chat", "docs", "status", "settings"].includes(urlTab)) {
        setTab(urlTab);
      }
    }
  }, []);

  const handleSetTab = useCallback((id: Tab) => {
    setTab(id);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("tab", id);
      window.history.replaceState({}, "", url);
    }
  }, []);

  // Gate the dashboard on a session. Ask the *backend* whether auth is on rather
  // than inferring it from the frontend env, so the two can never disagree.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await fetchAuthConfig();
        if (cancelled) return;
        if (!cfg.auth_enabled || !authConfigured) {
          setAuthReady(true);
          return;
        }
        const { data } = await supabase().auth.getSession();
        if (cancelled) return;
        if (!data.session) {
          router.replace("/login");
          return;
        }
        setAuthReady(true);
      } catch (err) {
        console.error("Backend unreachable:", err);
        if (!cancelled) setAuthReady(true); // Let it load into disconnected state instead of spinning
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  // Sessions live in localStorage, so hydrate after mount to avoid a mismatch.
  useEffect(() => {
    const { sessions: s, activeId: a } = loadSessions();
    setSessions(s);
    setActiveId(a);
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted) saveSessions(sessions, activeId);
  }, [sessions, activeId, mounted]);

  const refresh = useCallback(() => {
    api
      .status()
      .then(setStatus)
      .catch((err) => {
        // A dropped/expired session must send the user to sign in, not fail silently
        // and leave a dashboard that looks alive but answers nothing.
        if (err instanceof ApiError && err.isUnauthorized) router.replace("/login");
      });
  }, [router]);

  useEffect(() => {
    if (!authReady) return;
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, [refresh, authReady]);

  async function signOut() {
    if (authConfigured) await supabase().auth.signOut();
    router.replace("/login");
  }

  const active = sessions.find((s) => s.id === activeId);

  const updateActive = useCallback(
    (updater: (s: Session) => Session) =>
      setSessions((all) => all.map((s) => (s.id === activeId ? updater(s) : s))),
    [activeId],
  );

  function createSession() {
    const s = newSession();
    setSessions((all) => [s, ...all]);
    setActiveId(s.id);
    handleSetTab("chat");
  }

  function removeSession(id: string) {
    setSessions((all) => {
      const next = all.filter((s) => s.id !== id);
      if (next.length === 0) {
        const fresh = newSession();
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  }

  // Hold the shell back until we know whether a session is required, so a
  // signed-out visitor never sees a flash of the dashboard before redirecting.
  if (!authReady) {
    return (
      <div className="flex h-dvh items-center justify-center">
        <span className="sr-only">Checking your session…</span>
        <div className="border-muted border-t-primary size-6 animate-spin rounded-full border-2" />
      </div>
    );
  }

  return (
    <div className="flex h-dvh overflow-hidden">
      {/* Sidebar */}
      <nav className="bg-sidebar hidden w-[276px] shrink-0 flex-col gap-5 border-r px-5 py-6 md:flex">
        <div className="flex items-center gap-3 px-1">
          <div className="bg-primary text-primary-foreground flex size-9 shrink-0 items-center justify-center rounded-xl">
            <Network className="size-4" />
          </div>
          <div className="min-w-0">
            <h1 className="text-[15px] leading-none font-semibold tracking-tight">Aether AI</h1>
            <p className="text-muted-foreground mt-1 text-[11.5px] leading-none">Conversational RAG</p>
          </div>
        </div>

        <Button onClick={createSession} className="w-full">
          <Plus className="size-4" /> New session
        </Button>

        <div className="flex flex-col gap-1">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => handleSetTab(id)}
              aria-current={tab === id ? "page" : undefined}
              className={cn(
                "relative flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                tab === id
                  ? "bg-sidebar-accent text-sidebar-accent-foreground before:bg-primary font-semibold before:absolute before:top-1/2 before:-left-5 before:h-5 before:w-[3px] before:-translate-y-1/2 before:rounded-r"
                  : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground font-medium",
              )}
            >
              <Icon className="size-4" /> {label}
            </button>
          ))}
        </div>

        {/* Sessions */}
        <div className="flex min-h-0 flex-1 flex-col border-t pt-3">
          <div className="mb-2 flex items-center justify-between px-2">
            <p className="text-muted-foreground text-[11px] font-semibold tracking-wider uppercase">
              Sessions
            </p>
            <span className="text-muted-foreground tabular text-[11px]">{sessions.length}</span>
          </div>
          <ScrollArea className="-mx-1 min-h-0 flex-1 px-1">
            <div className="flex flex-col gap-0.5">
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={cn(
                    "group flex w-full items-center gap-2 rounded-lg px-2.5 py-2 transition-colors",
                    s.id === activeId ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
                  )}
                >
                  <button
                    onClick={() => { setActiveId(s.id); handleSetTab("chat"); }}
                    className="flex min-w-0 flex-1 flex-col text-left"
                  >
                    <span
                      className={cn(
                        "truncate text-[12.5px]",
                        s.id === activeId ? "text-primary font-semibold" : "font-medium",
                      )}
                    >
                      {s.title}
                    </span>
                    <span className="text-muted-foreground text-[10.5px]">
                      {s.messages.length ? relativeTime(s.updatedAt) : "empty"}
                    </span>
                  </button>
                  <button
                    onClick={() => removeSession(s.id)}
                    aria-label={`Delete session ${s.title}`}
                    className="text-muted-foreground hover:text-destructive shrink-0 opacity-0 transition group-hover:opacity-100 focus-visible:opacity-100"
                  >
                    <X className="size-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </ScrollArea>
          {sessions.length > 1 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => { const f = newSession(); setSessions([f]); setActiveId(f.id); }}
              className="text-muted-foreground hover:text-destructive mt-1 h-7 justify-start px-2 text-[11.5px]"
            >
              <Trash2 className="size-3.5" /> Clear all
            </Button>
          )}
        </div>

        <div className="flex flex-col gap-1 border-t pt-3">
          <div className="bg-secondary/60 mb-1 flex items-center gap-2 rounded-lg px-3 py-2">
            <span className="relative flex size-1.5 shrink-0">
              <span className="bg-success absolute inline-flex size-full animate-ping rounded-full opacity-60" />
              <span className="bg-success relative inline-flex size-1.5 rounded-full" />
            </span>
            <span className="text-muted-foreground truncate text-[11.5px] font-medium">
              {status?.llm_provider
                ? `${status.llm_provider} · ${status.llm_model}`
                : "connecting…"}
            </span>
          </div>
          {/* No tooltip here: the button carries a visible label already, and a
              tooltip that repeats it is noise for pointer and screen-reader users alike. */}
          <Button
            variant="ghost"
            onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
            className="text-muted-foreground w-full justify-start px-3"
          >
            {mounted && resolvedTheme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            <span className="text-[13px] font-medium">
              {mounted && resolvedTheme === "dark" ? "Light mode" : "Dark mode"}
            </span>
          </Button>

          {status?.auth_enabled && (
            <Button
              variant="ghost"
              onClick={signOut}
              className="text-muted-foreground hover:text-destructive w-full justify-start px-3"
            >
              <LogOut className="size-4" />
              <span className="truncate text-[13px] font-medium">
                {status.user_email ? `Sign out — ${status.user_email}` : "Sign out"}
              </span>
            </Button>
          )}
        </div>
      </nav>

      {/* Workspace */}
      <main className="min-w-0 flex-1 overflow-hidden">
        {mounted && active && (
          <>
            <div className={cn("h-full", tab !== "chat" && "hidden")}>
              <ChatView session={active} onUpdate={updateActive} />
            </div>
            <div className={cn("h-full", tab !== "docs" && "hidden")}>
              <DocumentsView status={status} onRefresh={refresh} />
            </div>
            <div className={cn("h-full", tab !== "status" && "hidden")}>
              <StatusView status={status} />
            </div>
            <div className={cn("h-full", tab !== "settings" && "hidden")}>
              <SettingsView status={status} onRefresh={refresh} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

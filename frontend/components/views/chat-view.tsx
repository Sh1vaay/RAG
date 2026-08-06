"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowUp, Bot, CornerDownLeft, FileText, Globe, Quote, Route, TriangleAlert, User,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Session } from "@/lib/sessions";
import { titleFrom } from "@/lib/sessions";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const SUGGESTIONS = [
  "Summarize the key takeaways from the documents.",
  "What are the three intake checks for green coffee?",
];

/** Minimal inline markdown — bold, italic, code. Escaping is inherent: we
 *  build React nodes rather than setting innerHTML. */
function RichText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g).filter(Boolean);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**"))
          return <strong key={i} className="font-semibold">{part.slice(2, -2)}</strong>;
        if (part.startsWith("`") && part.endsWith("`"))
          return (
            <code key={i} className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[0.85em] text-primary">
              {part.slice(1, -1)}
            </code>
          );
        if (part.startsWith("*") && part.endsWith("*"))
          return <em key={i}>{part.slice(1, -1)}</em>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function Avatar({ role }: { role: "user" | "assistant" }) {
  if (role === "user") {
    return (
      <div className="flex size-7 shrink-0 items-center justify-center rounded-lg border bg-secondary">
        <User className="size-3.5 text-muted-foreground" />
      </div>
    );
  }
  return (
    <div className="bg-primary text-primary-foreground flex size-7 shrink-0 items-center justify-center rounded-lg">
      <Bot className="size-4" />
    </div>
  );
}

interface Props {
  session: Session;
  onUpdate: (updater: (s: Session) => Session) => void;
}

export function ChatView({ session, onUpdate }: Props) {
  const [pending, setPending] = useState(false);
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [session.messages.length, pending]);

  const lastAnswer = [...session.messages].reverse().find((m) => m.role === "assistant" && !m.error);
  const sources = lastAnswer?.sources ?? [];

  async function send(text: string) {
    const message = text.trim();
    if (!message || pending) return;
    setDraft("");
    if (taRef.current) taRef.current.style.height = "auto";

    // Snapshot history *before* appending, so the new turn isn't duplicated.
    const history = session.messages
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

    onUpdate((s) => ({
      ...s,
      title: s.messages.length === 0 ? titleFrom(message) : s.title,
      updatedAt: Date.now(),
      messages: [...s.messages, { role: "user", content: message }],
    }));

    setPending(true);
    try {
      const data = await api.chat(message, history);
      onUpdate((s) => ({
        ...s,
        updatedAt: Date.now(),
        messages: [
          ...s.messages,
          {
            role: "assistant",
            content: data.answer,
            route: data.route,
            sources: data.sources,
            grounded: data.grounded !== false,
          },
        ],
      }));
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      onUpdate((s) => ({
        ...s,
        updatedAt: Date.now(),
        messages: [...s.messages, { role: "assistant", content: msg, error: true }],
      }));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex h-full gap-4 overflow-hidden p-4 md:p-6">
      <Card className="relative flex min-w-0 flex-1 flex-col gap-0 overflow-hidden py-0">
        <header className="flex shrink-0 items-center justify-between gap-3 border-b px-5 py-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <span className="relative flex size-1.5 shrink-0">
              <span className="bg-success absolute inline-flex size-full animate-ping rounded-full opacity-60" />
              <span className="bg-success relative inline-flex size-1.5 rounded-full" />
            </span>
            <span className="truncate text-sm font-medium">{session.title}</span>
          </div>
          <Badge variant="secondary" className="shrink-0 font-mono text-[10px] tracking-wider uppercase">
            {lastAnswer?.route ?? "ready"}
          </Badge>
        </header>

        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-6 px-5 py-7 pb-44 md:px-8">
            {session.messages.length === 0 && (
              <div className="flex max-w-3xl gap-3">
                <Avatar role="assistant" />
                <div className="bg-card min-w-0 flex-1 rounded-2xl rounded-tl-md border px-4 py-3.5 text-[0.925rem] leading-relaxed">
                  Ask anything about your indexed documents. Answers are grounded in retrieved
                  context and graded before they reach you.
                </div>
              </div>
            )}

            {session.messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="flex max-w-3xl flex-row-reverse gap-3 self-end">
                  <Avatar role="user" />
                  <div className="bg-primary text-primary-foreground max-w-[85%] rounded-2xl rounded-tr-md px-4 py-3 text-[0.925rem] leading-relaxed">
                    {m.content}
                  </div>
                </div>
              ) : (
                <div key={i} className="flex max-w-3xl gap-3">
                  <Avatar role="assistant" />
                  <div className="min-w-0 flex-1">
                    <span className="mb-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1">
                      {m.route && (
                        <span className="text-muted-foreground flex items-center gap-1 text-[10.5px] font-semibold tracking-wider uppercase">
                          <Route className="size-3" /> {m.route} route
                        </span>
                      )}
                      {/* An ungrounded answer that looks grounded is worse than no
                          answer, so say plainly where it came from. */}
                      {!m.error && m.grounded === false && (
                        <span className="text-warning border-warning/30 bg-warning/10 flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10.5px] font-semibold">
                          <Globe className="size-3" /> general knowledge — not your documents
                        </span>
                      )}
                    </span>
                    <div
                      className={cn(
                        "rounded-2xl rounded-tl-md border px-4 py-3.5 text-[0.925rem] leading-relaxed",
                        m.error ? "border-destructive/30 bg-destructive/5" : "bg-card",
                      )}
                    >
                      {m.error ? (
                        <div className="flex items-start gap-2">
                          <TriangleAlert className="text-destructive mt-0.5 size-4 shrink-0" />
                          <div>
                            <p className="text-destructive text-sm font-semibold">Request failed</p>
                            <p className="text-muted-foreground mt-1 text-sm">{m.content}</p>
                          </div>
                        </div>
                      ) : (
                        <RichText text={m.content} />
                      )}
                    </div>
                  </div>
                </div>
              ),
            )}

            {pending && (
              <div className="flex max-w-3xl gap-3">
                <Avatar role="assistant" />
                <div className="bg-card flex items-center gap-1.5 rounded-2xl rounded-tl-md border px-4 py-4">
                  {[0, 150, 300].map((d) => (
                    <span
                      key={d}
                      className="bg-primary/70 size-1.5 animate-bounce rounded-full"
                      style={{ animationDelay: `${d}ms` }}
                    />
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>

        <div className="absolute bottom-5 left-1/2 w-[calc(100%-2.5rem)] max-w-3xl -translate-x-1/2">
          <div className="scrollbar-none mb-2.5 flex gap-2 overflow-x-auto">
            {SUGGESTIONS.map((s) => (
              <Button
                key={s}
                variant="outline"
                size="sm"
                disabled={pending}
                onClick={() => send(s)}
                className="bg-card h-7 shrink-0 rounded-full text-xs font-medium"
              >
                {s.length > 34 ? `${s.slice(0, 34)}…` : s}
              </Button>
            ))}
          </div>
          <Card className="flex flex-row items-end gap-2 rounded-3xl p-1.5 shadow-lg">
            <Textarea
              ref={taRef}
              value={draft}
              rows={1}
              placeholder="Ask about your documents…"
              aria-label="Message"
              className="max-h-36 min-h-0 flex-1 resize-none border-0 bg-transparent px-3.5 py-2.5 text-[0.925rem] shadow-none focus-visible:ring-0 dark:bg-transparent"
              onChange={(e) => {
                setDraft(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 144)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(draft);
                }
              }}
            />
            <Button
              size="icon"
              aria-label="Send message"
              disabled={!draft.trim() || pending}
              onClick={() => send(draft)}
              className="mr-0.5 mb-0.5 size-9 shrink-0 rounded-full"
            >
              <ArrowUp className="size-4" />
            </Button>
          </Card>
          <p className="text-muted-foreground mt-2 flex items-center justify-center gap-1 text-[11px]">
            <CornerDownLeft className="size-3" /> Enter to send · Shift+Enter for a new line
          </p>
        </div>
      </Card>

      {/* Sources */}
      <Card className="hidden w-[310px] shrink-0 flex-col gap-0 overflow-hidden py-0 lg:flex">
        <header className="flex shrink-0 items-center gap-2 border-b px-5 py-3">
          <Quote className="text-muted-foreground size-4" />
          <h2 className="text-sm font-semibold">Sources</h2>
          {sources.length > 0 && (
            <Badge variant="secondary" className="ml-auto tabular">
              {sources.length}
            </Badge>
          )}
        </header>
        <ScrollArea className="min-h-0 flex-1">
          {sources.length === 0 ? (
            <div className="flex flex-col items-center px-6 pt-12 text-center">
              <Quote className="text-muted-foreground/50 mb-2 size-6" />
              <p className="text-muted-foreground text-xs">
                Citations appear here after a reply.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-2.5 p-3.5">
              {sources.map((s, i) => (
                <article
                  key={i}
                  className="bg-secondary/60 hover:border-primary/40 rounded-xl border p-3.5 transition-colors"
                >
                  <div className="mb-2 flex items-center gap-2">
                    <span className="bg-primary text-primary-foreground flex size-4.5 shrink-0 items-center justify-center rounded text-[10px] font-semibold">
                      {i + 1}
                    </span>
                    <span className="truncate text-xs font-semibold" title={s.title}>
                      {s.title}
                    </span>
                  </div>
                  <p className="text-muted-foreground line-clamp-4 text-[11.5px] leading-relaxed">
                    {s.snippet}
                  </p>
                  <span className="text-muted-foreground bg-card mt-2.5 inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10.5px] font-medium">
                    <FileText className="size-3" /> Page {s.page ?? "n/a"}
                  </span>
                </article>
              ))}
            </div>
          )}
        </ScrollArea>
      </Card>
    </div>
  );
}

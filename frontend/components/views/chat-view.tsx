"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowUp, Bot, Check, ChevronDown, Copy, CornerDownLeft, FileText, Globe, Quote,
  Route, TriangleAlert, User,
} from "lucide-react";
import { api, ApiError, type SourceDocument } from "@/lib/api";
import type { Session } from "@/lib/sessions";
import { titleFrom, type ChatMessage } from "@/lib/sessions";
import { pushMessage, pushSession } from "@/lib/chat-store";
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
/**
 * Renders an assistant reply as markdown.
 *
 * The previous version handled only inline markers and emitted everything else as
 * a <span>, so HTML collapsed newlines into spaces — paragraphs, lists and headings
 * all flattened into one block of text. That was the readability problem, not the
 * model's output.
 *
 * Raw HTML is deliberately NOT enabled (no rehype-raw): model output is untrusted
 * input, and react-markdown escapes it by default.
 */
function RichText({ text }: { text: string }) {
  return (
    <div className="space-y-3 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="leading-relaxed">{children}</p>,

          // Headings step down in size but stay close to body text — an answer is
          // not a document, and oversized headings break the reading rhythm.
          h1: ({ children }) => <h3 className="mt-4 text-[1.05em] font-semibold">{children}</h3>,
          h2: ({ children }) => <h3 className="mt-4 text-[1.02em] font-semibold">{children}</h3>,
          h3: ({ children }) => <h4 className="mt-3 font-semibold">{children}</h4>,
          h4: ({ children }) => <h4 className="mt-3 font-semibold">{children}</h4>,

          ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,

          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary font-medium underline underline-offset-2"
            >
              {children}
            </a>
          ),

          blockquote: ({ children }) => (
            <blockquote className="border-primary/40 text-muted-foreground border-l-2 pl-3 italic">
              {children}
            </blockquote>
          ),

          code: ({ className, children }) => {
            // react-markdown gives fenced blocks a `language-*` class; inline code
            // has none. Only the block form needs its own scroll container.
            const isBlock = Boolean(className);
            if (isBlock) {
              return (
                <code className="block overflow-x-auto font-mono text-[0.85em] leading-relaxed">
                  {children}
                </code>
              );
            }
            return (
              <code className="bg-muted text-primary rounded border px-1.5 py-0.5 font-mono text-[0.85em]">
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="bg-muted overflow-x-auto rounded-lg border p-3">{children}</pre>
          ),

          // Tables must scroll inside their own box, never widen the message bubble.
          table: ({ children }) => (
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full text-left text-[0.9em]">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="bg-muted border-b px-3 py-2 font-semibold">{children}</th>
          ),
          td: ({ children }) => <td className="border-b px-3 py-2 last:border-0">{children}</td>,

          hr: () => <hr className="border-border" />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

/**
 * A citation the reader can actually inspect.
 *
 * The snippet is clipped to four lines, which is often mid-sentence — so the card
 * expands to show it in full, along with the file it came from. It was previously
 * a static <article> that merely *looked* interactive: it had a hover border but
 * no click target, no keyboard access, and no way to read the rest of the text.
 */
function SourceCard({ index, source }: { index: number; source: SourceDocument }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  async function copy(e: React.MouseEvent) {
    e.stopPropagation(); // don't toggle the card while copying from it
    try {
      await navigator.clipboard.writeText(source.snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked (insecure origin or denied permission) */
    }
  }

  return (
    <div className="bg-secondary/60 hover:border-primary/40 focus-within:border-primary/60 rounded-xl border transition-colors">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-start gap-2 p-3.5 text-left"
      >
        <span className="bg-primary text-primary-foreground mt-0.5 flex size-[18px] shrink-0 items-center justify-center rounded text-[10px] font-semibold">
          {index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-semibold" title={source.title}>
            {source.title}
          </span>
          <span
            className={cn(
              "text-muted-foreground mt-1.5 block text-[11.5px] leading-relaxed",
              open ? "whitespace-pre-wrap" : "line-clamp-3",
            )}
          >
            {source.snippet}
          </span>
        </span>
        <ChevronDown
          className={cn(
            "text-muted-foreground mt-0.5 size-3.5 shrink-0 transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <div className="flex flex-wrap items-center gap-2 border-t px-3.5 py-2.5">
          {source.source?.startsWith("http") ? (
            <span className="text-muted-foreground bg-card inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10.5px] font-medium">
              <Globe className="size-3" /> Web Result
            </span>
          ) : (
            <span className="text-muted-foreground bg-card inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10.5px] font-medium">
              <FileText className="size-3" /> Page {source.page ?? "n/a"}
            </span>
          )}
          {source.source && (
            source.source.startsWith("http") ? (
              <a
                href={source.source}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:text-primary max-w-full truncate font-mono text-[10.5px] hover:underline"
                title={source.source}
              >
                {source.source}
              </a>
            ) : (
              <span
                className="text-muted-foreground max-w-full truncate font-mono text-[10.5px]"
                title={source.source}
              >
                {source.source}
              </span>
            )
          )}
          <button
            type="button"
            onClick={copy}
            className="text-muted-foreground hover:text-primary ml-auto inline-flex items-center gap-1 text-[10.5px] font-medium"
          >
            {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      )}
    </div>
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

type StreamMeta = { route: string; sources: SourceDocument[]; grounded: boolean };


export function ChatView({ session, onUpdate }: Props) {
  const [pending, setPending] = useState(false);
  const [draft, setDraft] = useState("");
  // Accumulates tokens during streaming; cleared once finalized into session.messages
  const [streamingContent, setStreamingContent] = useState("");
  const streamingRef = useRef("");   // stable ref so onDone callback reads the final value
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [session.messages.length, pending, streamingContent.length]);

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

    const isFirstTurn = session.messages.length === 0;
    const title = isFirstTurn ? titleFrom(message) : session.title;

    const userMessage: ChatMessage = { role: "user", content: message };
    onUpdate((s) => ({
      ...s,
      title: isFirstTurn ? title : s.title,
      updatedAt: Date.now(),
      messages: [...s.messages, userMessage],
    }));

    // Persist in the background — the UI already has the turn, and chat-store
    // silently degrades to local-only when Supabase is unavailable.
    void (async () => {
      if (isFirstTurn) await pushSession({ ...session, title });
      await pushMessage(session.id, userMessage);
    })();

    // Reset streaming state and start loading
    streamingRef.current = "";
    setStreamingContent("");
    setPending(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // finalMeta is populated by the onDone callback and read in the finally block.
    // Using a concrete type (not a local alias) ensures TS can narrow it correctly.
    let finalMeta: StreamMeta | null = null;

    try {
      await api.chatStream(
        message,
        history,
        // onToken — append each chunk and re-render
        (token) => {
          streamingRef.current += token;
          setStreamingContent(streamingRef.current);
        },
        // onDone — stash metadata; message is finalized in the finally block
        (meta) => {
          finalMeta = meta;
        },
        // onError — surface as an error message bubble
        (errMsg) => {
          const failure: ChatMessage = { role: "assistant", content: errMsg, error: true };
          onUpdate((s) => ({ ...s, updatedAt: Date.now(), messages: [...s.messages, failure] }));
          void pushMessage(session.id, failure);
        },
        ctrl.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const msg = err instanceof ApiError ? err.message : String(err);
        const failure: ChatMessage = { role: "assistant", content: msg, error: true };
        onUpdate((s) => ({ ...s, updatedAt: Date.now(), messages: [...s.messages, failure] }));
        void pushMessage(session.id, failure);
      }
    } finally {
      // Finalise the streamed content into the session message list
      if (streamingRef.current && finalMeta) {
        const meta = finalMeta as StreamMeta;
        const reply: ChatMessage = {
          role: "assistant",
          content: streamingRef.current,
          route: meta.route,
          sources: meta.sources,
          grounded: meta.grounded,
        };
        onUpdate((s) => ({ ...s, updatedAt: Date.now(), messages: [...s.messages, reply] }));
        void pushMessage(session.id, reply);
      }
      streamingRef.current = "";
      setStreamingContent("");
      setPending(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="flex h-full gap-4 overflow-hidden p-4 md:p-6">
      <Card className="flex min-w-0 flex-1 flex-col gap-0 overflow-hidden py-0">
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
          <div className="flex flex-col gap-6 px-5 py-7 md:px-8">
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
                          <Globe className="size-3" /> 
                          {m.sources && m.sources.length > 0 
                            ? "DuckDuckGo Web Search — not your documents" 
                            : "general knowledge — not your documents"}
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

            {/* Streaming bubble — visible while the LLM is generating */}
            {pending && streamingContent && (
              <div className="flex max-w-3xl gap-3">
                <Avatar role="assistant" />
                <div className="min-w-0 flex-1">
                  <div className="bg-card rounded-2xl rounded-tl-md border px-4 py-3.5 text-[0.925rem] leading-relaxed">
                    <RichText text={streamingContent} />
                    {/* Blinking cursor */}
                    <span
                      className="ml-0.5 inline-block h-[1.1em] w-0.5 translate-y-[2px] animate-pulse bg-current align-middle opacity-70"
                      aria-hidden
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Loading dots — only shown before the first token arrives */}
            {pending && !streamingContent && (
              <div className="flex max-w-3xl gap-3">
                <Avatar role="assistant" />
                <div className="bg-card flex items-center gap-3 rounded-2xl rounded-tl-md border px-4 py-3.5">
                  <div className="flex items-center gap-1.5 pt-0.5">
                    {[0, 150, 300].map((d) => (
                      <span
                        key={d}
                        className="bg-primary/70 size-1.5 animate-bounce rounded-full"
                        style={{ animationDelay: `${d}ms` }}
                      />
                    ))}
                  </div>
                  <span className="text-muted-foreground animate-pulse text-[0.925rem]">
                    Thinking or searching the web...
                  </span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>

        {/* A normal flex child, not absolutely positioned. The composer grows as the
            textarea does (up to max-h-36), and the scroll area above shrinks to match,
            so messages can never end up underneath it. The previous version floated
            this over the list and reserved a fixed pb-44 — which the composer outgrows
            as soon as the textarea expands. */}
        <div className="bg-card/80 shrink-0 border-t px-5 pt-3 pb-4 backdrop-blur md:px-8">
          <div className="mx-auto w-full max-w-3xl">
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
                <SourceCard key={i} index={i} source={s} />
              ))}
            </div>
          )}
        </ScrollArea>
      </Card>
    </div>
  );
}

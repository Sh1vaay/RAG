"use client";

import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, Loader2, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError, type ProviderCatalog, type StatusResponse } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

/** env var -> [state key, visible label, placeholder] */
const CREDENTIALS: Record<string, [string, string, string]> = {
  OPENAI_API_KEY: ["openai_key", "OpenAI API key", "sk-…"],
  ANTHROPIC_API_KEY: ["anthropic_key", "Anthropic API key", "sk-ant-…"],
  GOOGLE_API_KEY: ["google_key", "Google AI API key", "AIza…"],
  XAI_API_KEY: ["xai_key", "xAI API key", "xai-…"],
  COHERE_API_KEY: ["cohere_key", "Cohere API key", "Your Cohere key"],
};

interface Props {
  status: StatusResponse | null;
  onRefresh: () => void;
}

export function SettingsView({ status, onRefresh }: Props) {
  const [catalog, setCatalog] = useState<ProviderCatalog | null>(null);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<{ kind: "ok" | "warn" | "err"; text: string } | null>(null);

  const [llmProvider, setLlmProvider] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [embProvider, setEmbProvider] = useState("");
  const [embModel, setEmbModel] = useState("");
  const [routing, setRouting] = useState("semantic");
  const [reranker, setReranker] = useState("flashrank");
  const [keys, setKeys] = useState<Record<string, string>>({});

  useEffect(() => {
    api.providers().then(setCatalog).catch(() => setCatalog(null));
  }, []);

  // Seed the form from server state once, and whenever the user hits Reset.
  const reset = useMemo(
    () => () => {
      if (!status) return;
      setLlmProvider(status.llm_provider ?? "ollama");
      setEmbProvider(status.embedding_provider ?? "ollama");
      setLlmModel(status.llm_model ?? "");
      setEmbModel(status.embedding_model ?? "");
      setRouting(status.routing_method);
      setReranker(status.reranker_provider);
      setKeys({});
      setResult(null);
    },
    [status],
  );

  useEffect(() => {
    if (status && !llmProvider) reset();
  }, [status, llmProvider, reset]);

  const llmEntry = catalog?.chat.find((p) => p.id === llmProvider);
  const embEntry = catalog?.embedding.find((p) => p.id === embProvider);

  // Only ask for the credentials the current selection actually needs.
  const needed = useMemo(() => {
    const map = new Map<string, boolean>();
    [llmEntry, embEntry].forEach((p) => {
      if (p?.requires_key && p.key_env) map.set(p.key_env, p.key_configured);
    });
    if (reranker === "cohere") {
      const c = catalog?.embedding.find((p) => p.id === "cohere");
      map.set("COHERE_API_KEY", c?.key_configured ?? false);
    }
    return map;
  }, [llmEntry, embEntry, reranker, catalog]);

  const embeddingWillChange =
    status &&
    (embProvider !== status.embedding_provider || embModel !== (status.embedding_model ?? ""));

  async function save() {
    setSaving(true);
    setResult(null);
    try {
      const payload = {
        routing_method: routing,
        reranker_provider: reranker,
        llm_provider: llmProvider,
        llm_model: llmModel,
        embedding_provider: embProvider,
        embedding_model: embModel,
        ...Object.fromEntries(
          Object.entries(CREDENTIALS).map(([env, [field]]) => [
            field,
            needed.has(env) && keys[field]?.trim() ? keys[field].trim() : null,
          ]),
        ),
      };
      const data = await api.config(payload);
      setResult({ kind: data.embedding_changed ? "warn" : "ok", text: data.message });
      toast.success("Configuration applied");
      setKeys({});
      api.providers().then(setCatalog).catch(() => {});
      onRefresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setResult({ kind: "err", text: msg });
      toast.error("Configuration failed", { description: msg });
    } finally {
      setSaving(false);
    }
  }

  const label = (id: string, requiresKey: boolean) =>
    `${id.charAt(0).toUpperCase()}${id.slice(1)}${requiresKey ? "" : " (local)"}`;

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex max-w-3xl flex-col gap-6 p-6 md:p-10">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
          <p className="text-muted-foreground mt-1.5 text-sm">
            Choose which services run the pipeline. Applied immediately — no restart.
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Language model</CardTitle>
            <CardDescription>Runs generation, grading, rewriting and routing.</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-5 md:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="llm-provider">Provider</Label>
              <Select value={llmProvider} onValueChange={(v) => setLlmProvider(v ?? "")}>
                <SelectTrigger id="llm-provider" className="w-full">
                  <SelectValue placeholder="Select a provider" />
                </SelectTrigger>
                <SelectContent>
                  {catalog?.chat.map((p) => (
                    <SelectItem key={p.id} value={p.id}>{label(p.id, p.requires_key)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="llm-model">
                Model <span className="text-muted-foreground font-normal">(blank = default)</span>
              </Label>
              <Input
                id="llm-model"
                className="font-mono text-sm"
                placeholder={llmEntry?.default_model ?? ""}
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Embeddings</CardTitle>
            <CardDescription>
              Powers the index, semantic chunker and router.
              {catalog?.chat_only.length ? (
                <> {catalog.chat_only.join(" and ")} publish no embeddings API, so they are chat-only.</>
              ) : null}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-5">
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
              <div className="flex flex-col gap-2">
                <Label htmlFor="emb-provider">Provider</Label>
                <Select value={embProvider} onValueChange={(v) => setEmbProvider(v ?? "")}>
                  <SelectTrigger id="emb-provider" className="w-full">
                    <SelectValue placeholder="Select a provider" />
                  </SelectTrigger>
                  <SelectContent>
                    {catalog?.embedding.map((p) => (
                      <SelectItem key={p.id} value={p.id}>{label(p.id, p.requires_key)}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="emb-model">
                  Model <span className="text-muted-foreground font-normal">(blank = default)</span>
                </Label>
                <Input
                  id="emb-model"
                  className="font-mono text-sm"
                  placeholder={embEntry?.default_model ?? ""}
                  value={embModel}
                  onChange={(e) => setEmbModel(e.target.value)}
                />
              </div>
            </div>
            {embeddingWillChange && (
              <Alert>
                <TriangleAlert className="text-warning size-4" />
                <AlertTitle>This will invalidate the current index</AlertTitle>
                <AlertDescription>
                  Stored vectors belong to the old model&apos;s space. Re-ingest your documents
                  after applying, or queries will fail.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Pipeline</CardTitle>
            <CardDescription>Routing strategy and context reranking.</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-5 md:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="routing">Routing method</Label>
              <Select value={routing} onValueChange={(v) => setRouting(v ?? "semantic")}>
                <SelectTrigger id="routing" className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="semantic">Semantic embeddings</SelectItem>
                  <SelectItem value="llm">LLM structured</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-muted-foreground text-[11.5px]">
                LLM routing needs a capable model; small local models return invalid routes.
              </p>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="reranker">Reranker</Label>
              <Select value={reranker} onValueChange={(v) => setReranker(v ?? "flashrank")}>
                <SelectTrigger id="reranker" className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="flashrank">Flashrank (local)</SelectItem>
                  <SelectItem value="cohere">Cohere API</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-muted-foreground text-[11.5px]">
                Flashrank runs on CPU and needs no key.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Credentials</CardTitle>
            <CardDescription>
              Only the keys your selected services need. Held in the server process, never
              written to disk.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {needed.size === 0 ? (
              <p className="text-muted-foreground flex items-center gap-2 text-sm">
                <CheckCircle2 className="text-success size-4" />
                No credentials needed — everything selected runs locally.
              </p>
            ) : (
              [...needed.entries()].map(([env, configured]) => {
                const [field, labelText, placeholder] = CREDENTIALS[env];
                return (
                  <div key={env} className="flex flex-col gap-2">
                    <Label htmlFor={field}>
                      {labelText}
                      <span
                        className={
                          configured ? "text-success text-[11px] font-medium" : "text-warning text-[11px] font-medium"
                        }
                      >
                        {configured ? "already set" : "required"}
                      </span>
                    </Label>
                    <Input
                      id={field}
                      type="password"
                      autoComplete="off"
                      placeholder={configured ? "Leave blank to keep current key" : placeholder}
                      value={keys[field] ?? ""}
                      onChange={(e) => setKeys((k) => ({ ...k, [field]: e.target.value }))}
                    />
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>

        {result && (
          <Alert variant={result.kind === "err" ? "destructive" : "default"}>
            {result.kind === "ok" ? (
              <CheckCircle2 className="text-success size-4" />
            ) : (
              <TriangleAlert className={result.kind === "warn" ? "text-warning size-4" : "size-4"} />
            )}
            <AlertTitle>
              {result.kind === "ok" ? "Applied" : result.kind === "warn" ? "Applied — action needed" : "Failed"}
            </AlertTitle>
            <AlertDescription>{result.text}</AlertDescription>
          </Alert>
        )}

        <div className="flex justify-end gap-3 pb-4">
          <Button variant="outline" onClick={reset} disabled={saving}>Reset</Button>
          <Button onClick={save} disabled={saving || !llmProvider}>
            {saving && <Loader2 className="size-4 animate-spin" />}
            {saving ? "Applying…" : "Apply configuration"}
          </Button>
        </div>
      </div>
    </ScrollArea>
  );
}

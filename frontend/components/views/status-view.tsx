"use client";

import { AlertCircle, Activity, CheckCircle2, Cloud, Database, Layers, Split } from "lucide-react";
import type { StatusResponse } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";

/** Credential state is icon + word, never colour alone — with a green brand a
 *  green "ok" would otherwise be indistinguishable from ordinary chrome. */
function KeyState({ configured, required }: { configured?: boolean; required: boolean }) {
  if (!required)
    return (
      <span className="text-success inline-flex items-center gap-1 text-sm font-medium">
        <CheckCircle2 className="size-3.5" /> Not required
      </span>
    );
  return configured ? (
    <span className="text-success inline-flex items-center gap-1 text-sm font-medium">
      <CheckCircle2 className="size-3.5" /> Configured
    </span>
  ) : (
    <span className="text-destructive inline-flex items-center gap-1 text-sm font-medium">
      <AlertCircle className="size-3.5" /> Missing
    </span>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number | undefined;
}) {
  return (
    <Card className="gap-0 py-5">
      <CardContent className="px-5">
        <div className="text-muted-foreground mb-2.5 flex items-center gap-2">
          <Icon className="size-4" />
          <span className="text-[11.5px] font-semibold tracking-wider uppercase">{label}</span>
        </div>
        {value === undefined ? (
          <Skeleton className="h-7 w-20" />
        ) : (
          <p className="tabular truncate text-xl font-semibold tracking-tight">{value}</p>
        )}
      </CardContent>
    </Card>
  );
}

function ProviderCard({
  title,
  description,
  provider,
  model,
  keyConfigured,
}: {
  title: string;
  description: string;
  provider?: string;
  model?: string;
  keyConfigured?: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="flex flex-col gap-3.5">
          <div className="flex items-center justify-between gap-4">
            <dt className="text-muted-foreground text-sm">Provider</dt>
            <dd className="text-sm font-semibold capitalize">{provider ?? "—"}</dd>
          </div>
          <div className="flex items-center justify-between gap-4">
            <dt className="text-muted-foreground text-sm">Model</dt>
            <dd className="max-w-[190px] truncate font-mono text-xs" title={model}>
              {model ?? "—"}
            </dd>
          </div>
          <div className="flex items-center justify-between gap-4">
            <dt className="text-muted-foreground text-sm">Credential</dt>
            <dd>
              <KeyState configured={keyConfigured} required={Boolean(provider && provider !== "ollama")} />
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}

const STAGES = [
  ["Analyze", "Pydantic filters"],
  ["Retrieve", "FAISS + BM25 · RRF"],
  ["Rerank", "Flashrank CPU"],
  ["Generate", "Graded for grounding"],
];

export function StatusView({ status }: { status: StatusResponse | null }) {
  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex max-w-[1400px] flex-col gap-7 p-6 md:p-10">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">Status</h1>
          <p className="text-muted-foreground mt-1.5 max-w-2xl text-sm">
            Live view of the index and the active providers.
          </p>
        </header>

        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <Stat icon={Database} label="Index" value={status ? (status.database_loaded ? "Active" : "Empty") : undefined} />
          <Stat icon={Layers} label="Chunks" value={status?.document_chunks} />
          <Stat icon={Split} label="Routing" value={status ? (status.routing_method === "semantic" ? "Semantic" : "LLM") : undefined} />
          <Stat icon={Activity} label="Tracing" value={status ? (status.langsmith_tracing ? "On" : "Off") : undefined} />
          <Stat icon={Cloud} label="Cloud Storage" value={status ? (status.storage_enabled ? "Active" : "Local only") : undefined} />
        </div>

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          <ProviderCard
            title="Language model"
            description="Generation, grading and routing"
            provider={status?.llm_provider}
            model={status?.llm_model}
            keyConfigured={status?.llm_key_configured}
          />
          <ProviderCard
            title="Embeddings"
            description="Index, chunker and semantic router"
            provider={status?.embedding_provider}
            model={status?.embedding_model}
            keyConfigured={status?.embedding_key_configured}
          />
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Retrieval pipeline</CardTitle>
            <CardDescription>
              Stages a heavy-path query passes through, in order.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Numbered because this genuinely is a sequence — each stage consumes
                the previous stage's output. */}
            <ol className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              {STAGES.map(([title, desc], i) => (
                <li key={title} className="bg-secondary/50 rounded-xl border p-4">
                  <span className="text-muted-foreground font-mono text-[11px]">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <h3 className="mt-1.5 text-sm font-semibold">{title}</h3>
                  <p className="text-muted-foreground mt-1 text-[11.5px]">{desc}</p>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
}

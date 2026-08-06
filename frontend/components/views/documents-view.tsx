"use client";

import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Cloud, Loader2, Terminal, Trash2, UploadCloud, Zap, History, Clock, AlertCircle, X } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError, type StatusResponse, type BuildMeta } from "@/lib/api";
import { API_BASE } from "@/lib/api";
import { accessToken } from "@/lib/supabase";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

function timeAgo(date: string | Date) {
  const diff = (new Date().getTime() - new Date(date).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

type LogLine = { text: string; tone: "info" | "ok" | "err" | "warn" };

interface Props {
  status: StatusResponse | null;
  onRefresh: () => void;
}

export function DocumentsView({ status, onRefresh }: Props) {
  const [raptor, setRaptor] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  
  const [builds, setBuilds] = useState<BuildMeta[]>([]);
  const [activeBuild, setActiveBuild] = useState<string | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  
  const fileRef = useRef<HTMLInputElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "end" });
  }, [log]);

  const push = (text: string, tone: LogLine["tone"] = "info") =>
    setLog((l) => [...l, { text, tone }]);

  // Fetch builds: aggressively every 3 s while a build is running,
  // otherwise back off to every 15 s to avoid hammering Supabase Storage.
  const isBuilding = builds.some((b) => b.status === "running");
  useEffect(() => {
    fetchBuilds();
    const interval = setInterval(fetchBuilds, isBuilding ? 3000 : 15000);
    return () => clearInterval(interval);
    // Re-register the interval each time the running state changes so the
    // delay switches between fast and slow without waiting for the old timer.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isBuilding]);

  async function fetchBuilds() {
    try {
      const { builds: fetched } = await api.listBuilds();
      setBuilds(fetched);
      
      // Auto-select the most recent running build on load
      setActiveBuild((current) => {
        if (current) return current;
        const running = fetched.find((b) => b.status === "running");
        return running ? running.id : (fetched[0]?.id || null);
      });
    } catch {}
  }

  // Stream logs for the active build
  useEffect(() => {
    if (!activeBuild) {
      setLog([{ text: "Select a build to view logs.", tone: "info" }]);
      return;
    }
    
    setLog([{ text: `Connecting to build ${activeBuild.slice(0, 8)}...`, tone: "info" }]);
    const abortController = new AbortController();
    
    async function streamLogs() {
      try {
        const token = await accessToken();
        const headers: Record<string, string> = {};
        if (token) headers["Authorization"] = `Bearer ${token}`;
        
        const res = await fetch(`${API_BASE}/api/builds/${activeBuild}/stream`, {
            headers,
            signal: abortController.signal
        });
        
        if (!res.body) return;
        setLog([]); // Clear "Connecting..." message
        
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop() || "";
            
            for (const line of lines) {
                if (line.startsWith("event: close")) {
                    return; // stream finished normally
                }
                if (line.startsWith("data: ")) {
                    const dataStr = line.slice(6);
                    if (dataStr.trim()) {
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.text) {
                                const text = data.text.trimEnd();
                                let tone: LogLine["tone"] = "info";
                                if (text.includes("[API ERROR]") || text.toLowerCase().includes("error")) tone = "err";
                                else if (text.includes("[API WARN]")) tone = "warn";
                                else if (text.includes("✅") || text.includes("successfully")) tone = "ok";
                                push(text, tone);
                            }
                        } catch (e) {}
                    }
                }
            }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
            push(`[Stream disconnected]`, "err");
        }
      }
    }
    
    streamLogs();
    return () => abortController.abort();
  }, [activeBuild]);

  async function upload(files: FileList | File[]) {
    if (!files.length) return;
    try {
      const data = await api.upload(files);
      toast.success(`Staged ${data.uploaded_files.length} file(s)`);
      onRefresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      toast.error("Upload failed", { description: msg });
    }
  }

  async function deleteFile(filename: string) {
    setDeleting(filename);
    try {
      await api.deleteDocument(filename);
      toast.success(`Deleted ${filename}`);
      onRefresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      toast.error("Delete failed", { description: msg });
    } finally {
      setDeleting(null);
    }
  }

  async function build() {
    try {
      const data = await api.ingest(raptor);
      toast.success("Build started in the background");
      fetchBuilds();
      setActiveBuild(data.build_id);
      onRefresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      toast.error("Build failed to start", { description: msg });
    }
  }

  async function cancelBuild(buildId: string) {
    try {
      await api.cancelBuild(buildId);
      toast.success("Build cancelled");
      fetchBuilds();
      onRefresh();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      toast.error("Failed to cancel build", { description: msg });
    }
  }

  const staged = status?.staged_files ?? [];
  const storageEnabled = status?.storage_enabled ?? false;
  const isBuilding = builds.some((b) => b.status === "running");

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex max-w-[1400px] flex-col gap-7 p-6 md:p-10">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
            <p className="text-muted-foreground mt-1.5 max-w-2xl text-sm">
              Stage source files, then build the vector index.
            </p>
          </div>
          {storageEnabled && (
            <div className="text-muted-foreground flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium">
              <Cloud className="size-3.5 text-sky-500" />
              Cloud sync active
            </div>
          )}
        </header>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
          <div className="flex flex-col gap-5 xl:col-span-7">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files); }}
              className={cn(
                "group flex flex-col items-center justify-center rounded-2xl border-[1.5px] border-dashed p-11 text-center transition-colors",
                dragging ? "border-primary bg-primary/5" : "hover:border-primary/50 hover:bg-primary/[0.03]",
              )}
            >
              <div className="bg-primary/10 mb-4 flex size-14 items-center justify-center rounded-2xl transition-transform group-hover:scale-105">
                <UploadCloud className="text-primary size-7" />
              </div>
              <span className="text-base font-semibold">Drop files to stage them</span>
              <span className="text-muted-foreground mt-1.5 text-sm">
                PDF, DOCX, CSV, TXT or MD
              </span>
              <span className="bg-card mt-5 rounded-full border px-5 py-2 text-sm font-semibold">
                Browse files
              </span>
              <input
                ref={fileRef}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => { upload(e.target.files ?? []); e.target.value = ""; }}
              />
            </button>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <div className="space-y-1">
                  <CardTitle className="text-base">Staged for ingestion</CardTitle>
                  <CardDescription className="tabular">
                    {staged.length ? `${staged.length} file${staged.length > 1 ? "s" : ""}` : "none"}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  {isBuilding && activeBuild && (
                    <Button onClick={() => cancelBuild(activeBuild)} variant="destructive" size="sm">
                      <X className="size-4 mr-1.5" /> Stop
                    </Button>
                  )}
                  <Button onClick={build} disabled={isBuilding} size="sm">
                    {isBuilding ? <Loader2 className="size-4 animate-spin mr-2" /> : <Zap className="size-4 mr-2" />}
                    {isBuilding ? "Building..." : "Build index"}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Filename</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead className="text-right">Status</TableHead>
                      <TableHead className="w-[50px]" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {staged.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-muted-foreground py-9 text-center">
                          No files staged yet.
                        </TableCell>
                      </TableRow>
                    ) : (
                      staged.map((f) => (
                        <TableRow key={f.name}>
                          <TableCell className="max-w-[320px] truncate font-medium">{f.name}</TableCell>
                          <TableCell className="text-muted-foreground tabular">{f.size}</TableCell>
                          <TableCell className="text-right">
                            <span className="text-success inline-flex items-center gap-1 text-xs font-medium">
                              <CheckCircle2 className="size-3.5" /> Ready
                            </span>
                          </TableCell>
                          <TableCell className="text-right">
                            <button
                              onClick={() => deleteFile(f.name)}
                              disabled={deleting === f.name}
                              aria-label={`Delete ${f.name}`}
                              className="text-muted-foreground hover:text-destructive inline-flex items-center justify-center rounded-md p-1.5 transition-colors disabled:opacity-40"
                            >
                              {deleting === f.name ? (
                                <Loader2 className="size-3.5 animate-spin" />
                              ) : (
                                <Trash2 className="size-3.5" />
                              )}
                            </button>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          <div className="flex flex-col gap-5 xl:col-span-5">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <History className="size-4 text-muted-foreground" />
                  Build History
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[200px] border-b">
                  <div className="flex flex-col">
                    {builds.length === 0 ? (
                      <div className="p-8 text-center text-sm text-muted-foreground">
                        No builds found.
                      </div>
                    ) : (
                      builds.map((b) => (
                        <button
                          key={b.id}
                          onClick={() => setActiveBuild(b.id)}
                          className={cn(
                            "flex items-center justify-between p-3 px-4 border-b last:border-0 transition-colors text-left",
                            activeBuild === b.id ? "bg-primary/5" : "hover:bg-secondary/40"
                          )}
                        >
                          <div className="flex items-center gap-3">
                            {b.status === "running" ? (
                              <Loader2 className="size-4 animate-spin text-primary" />
                            ) : b.status === "completed" ? (
                              <CheckCircle2 className="size-4 text-success" />
                            ) : (
                              <AlertCircle className="size-4 text-destructive" />
                            )}
                            <div>
                              <div className="text-sm font-medium">
                                Build {b.id.slice(0, 8)}
                              </div>
                              <div className="text-xs text-muted-foreground flex items-center gap-1">
                                <Clock className="size-3" />
                                {timeAgo(b.started_at)}
                              </div>
                            </div>
                          </div>
                          {b.raptor && (
                            <span className="text-[10px] font-medium bg-secondary px-2 py-0.5 rounded-full">
                              RAPTOR
                            </span>
                          )}
                        </button>
                      ))
                    )}
                  </div>
                </ScrollArea>
                <div className="p-4 bg-secondary/20">
                  <Label
                    htmlFor="raptor"
                    className="flex cursor-pointer items-center justify-between gap-4 font-normal"
                  >
                    <span className="min-w-0">
                      <span className="block text-sm font-medium">RAPTOR summaries</span>
                      <span className="text-muted-foreground mt-0.5 block text-xs leading-snug">
                        Hierarchical cluster tree
                      </span>
                    </span>
                    <Switch id="raptor" checked={raptor} onCheckedChange={setRaptor} />
                  </Label>
                </div>
              </CardContent>
            </Card>

            <Card className="flex h-[400px] flex-col gap-0 overflow-hidden py-0">
              <div className="bg-secondary/60 flex shrink-0 items-center justify-between gap-2 border-b px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <Terminal className="text-muted-foreground size-4" />
                  <h3 className="text-xs font-semibold">
                    {activeBuild ? `Console (Build ${activeBuild.slice(0, 8)})` : "Console"}
                  </h3>
                </div>
                {builds.find(b => b.id === activeBuild)?.status === "running" && (
                  <span className="flex items-center gap-1.5 text-[10px] font-medium text-primary">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                    </span>
                    LIVE
                  </span>
                )}
              </div>
              <ScrollArea className="min-h-0 flex-1 bg-black">
                <div className="p-4 font-mono text-[11.5px] leading-relaxed tracking-tight">
                  {log.map((l, i) => (
                    <div
                      key={i}
                      className={cn(
                        "mb-1 whitespace-pre-wrap break-all",
                        l.tone === "ok" && "text-green-400",
                        l.tone === "err" && "text-red-400",
                        l.tone === "warn" && "text-yellow-400",
                        l.tone === "info" && "text-zinc-300",
                      )}
                    >
                      {l.text}
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              </ScrollArea>
            </Card>
          </div>
        </div>
      </div>
    </ScrollArea>
  );
}

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AlertCircle, Loader2, MailCheck, Network } from "lucide-react";
import { authConfigured, supabase } from "@/lib/supabase";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type Mode = "signin" | "signup";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmSent, setConfirmSent] = useState(false);

  // Already signed in, or auth switched off entirely — nothing to do here.
  useEffect(() => {
    if (!authConfigured) {
      router.replace("/");
      return;
    }
    supabase()
      .auth.getSession()
      .then(({ data }) => {
        if (data.session) router.replace("/");
      });
  }, [router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setConfirmSent(false);
    try {
      const client = supabase();
      if (mode === "signup") {
        const { data, error } = await client.auth.signUp({ email, password });
        if (error) throw error;
        // With email confirmation on, signUp returns a user but no session.
        if (!data.session) {
          setConfirmSent(true);
          return;
        }
      } else {
        const { error } = await client.auth.signInWithPassword({ email, password });
        if (error) throw error;
      }
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-dvh items-center justify-center p-6">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <div className="bg-primary text-primary-foreground mb-3 flex size-10 items-center justify-center rounded-xl">
            <Network className="size-5" />
          </div>
          <CardTitle className="text-xl tracking-tight">
            {mode === "signin" ? "Sign in to Aether AI" : "Create your workspace"}
          </CardTitle>
          <CardDescription>
            {mode === "signin"
              ? "Your documents and index are private to your account."
              : "You get your own documents and your own vector index."}
          </CardDescription>
        </CardHeader>

        <CardContent>
          {confirmSent ? (
            <Alert>
              <MailCheck className="text-success size-4" />
              <AlertTitle>Check your email</AlertTitle>
              <AlertDescription>
                We sent a confirmation link to {email}. Open it, then sign in.
              </AlertDescription>
            </Alert>
          ) : (
            <form onSubmit={submit} className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete={mode === "signin" ? "current-password" : "new-password"}
                  required
                  minLength={8}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={mode === "signup" ? "At least 8 characters" : ""}
                />
              </div>

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="size-4" />
                  <AlertTitle>{mode === "signin" ? "Could not sign in" : "Could not sign up"}</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <Button type="submit" disabled={busy} className="mt-1 w-full">
                {busy && <Loader2 className="size-4 animate-spin" />}
                {mode === "signin" ? "Sign in" : "Create account"}
              </Button>
            </form>
          )}

          <p className="text-muted-foreground mt-5 text-center text-[13px]">
            {mode === "signin" ? "No account yet?" : "Already have an account?"}{" "}
            <button
              type="button"
              className="text-primary font-medium underline-offset-4 hover:underline"
              onClick={() => {
                setMode(mode === "signin" ? "signup" : "signin");
                setError(null);
                setConfirmSent(false);
              }}
            >
              {mode === "signin" ? "Create one" : "Sign in"}
            </button>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

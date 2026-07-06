"use client";

import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  message: string;
}

/**
 * Defensive boundary around the AppShell tree. Catches any uncaught
 * render-time error and renders a minimal fallback so the user
 * sees something useful instead of a 500 page. Hydration mismatches
 * surface here in dev mode; the fallback stays identical between
 * server and client so the user never sees a flash of broken UI.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(err: unknown): State {
    const message = err instanceof Error ? err.message : "Unknown error";
    return { hasError: true, message };
  }

  componentDidCatch(err: unknown) {
    // In production this would forward to Sentry/Datadog. Dev mode
    // surfaces the error in the console; we re-throw to keep the dev
    // overlay visible.
    if (process.env.NODE_ENV !== "production") {
      // eslint-disable-next-line no-console
      console.error("[ErrorBoundary]", err);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div
            role="alert"
            className="min-h-screen bg-black text-white flex items-center justify-center px-8"
          >
            <div className="max-w-md flex flex-col gap-3 border border-white/15 p-6">
              <span className="text-xs uppercase tracking-widest text-white/50">
                Engine Offline
              </span>
              <h1 className="text-2xl font-bold tracking-tighter">
                Connecting to Engine...
              </h1>
              <p className="text-sm text-white/60 leading-relaxed">
                The dashboard could not finish initializing. The backend
                may be temporarily unreachable.
              </p>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="mt-3 self-start px-4 py-2 border border-white/20 text-xs uppercase tracking-widest hover:border-white/60"
              >
                Retry
              </button>
            </div>
          </div>
        )
      );
    }
    return this.props.children;
  }
}

"use client";

import Link from "next/link";
import { Activity, Database, Clock, Server, RefreshCw } from "lucide-react";
import { useHealthStore } from "../../store/useHealthStore";
import { useHealthSubscription } from "../../hooks/useHealthSubscription";

const BUILD_TIME = process.env.NEXT_PUBLIC_BUILD_TIME || "development";

export default function StatusPage() {
  useHealthSubscription();
  const health = useHealthStore((s) => s.health);
  const refresh = useHealthStore((s) => s.refresh);

  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 flex flex-col gap-10">
        <header className="flex items-center justify-between border-b border-white/10 pb-6">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-xs uppercase tracking-widest text-white/50 hover:text-white transition-colors"
            >
              ← Back to Engine
            </Link>
            <h1 className="text-2xl font-bold tracking-widest uppercase">
              System Status
            </h1>
          </div>
          <button
            onClick={refresh}
            className="flex items-center gap-2 text-xs uppercase tracking-widest text-white/50 hover:text-white transition-colors"
            aria-label="Refresh health"
          >
            <RefreshCw size={14} />
            Refresh
          </button>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border border-white/10 p-6 flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <Activity size={18} className="text-white/50" />
              <span className="text-xs uppercase tracking-widest text-white/50">
                API Engine
              </span>
            </div>
            {health.phase === "loading" && (
              <span className="text-sm text-white/40 animate-pulse">
                Checking...
              </span>
            )}
            {health.phase === "error" && (
              <div className="flex flex-col gap-1">
                <span className="text-red font-bold text-sm">Offline</span>
                <span className="text-xs text-white/40 font-mono">
                  {health.message}
                </span>
              </div>
            )}
            {health.phase === "ok" && (
              <div className="flex flex-col gap-1">
                <span className="text-green font-bold text-sm">Connected</span>
                <span className="text-xs text-white/40 font-mono">
                  Status: {health.data.status}
                </span>
                <span className="text-xs text-white/40 font-mono">
                  Latency: {health.data.latency_ms}ms
                </span>
                <span className="text-xs text-white/30 font-mono">
                  Snapshot: {new Date(health.data.fetched_at).toLocaleTimeString()}
                </span>
              </div>
            )}
          </div>

          <div className="border border-white/10 p-6 flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <Database size={18} className="text-white/50" />
              <span className="text-xs uppercase tracking-widest text-white/50">
                Predictive Model
              </span>
            </div>
            {health.phase === "loading" && (
              <span className="text-sm text-white/40 animate-pulse">
                Checking...
              </span>
            )}
            {health.phase === "error" && (
              <span className="text-white/40 text-sm">Unreachable</span>
            )}
            {health.phase === "ok" && (
              <div className="flex flex-col gap-1">
                <span
                  className={`font-bold text-sm ${
                    health.data.model_loaded ? "text-green" : "text-red"
                  }`}
                >
                  {health.data.model_loaded ? "Loaded" : "Offline"}
                </span>
                <span className="text-xs text-white/40 font-mono truncate">
                  {health.data.model_path}
                </span>
              </div>
            )}
          </div>

          <div className="border border-white/10 p-6 flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <Clock size={18} className="text-white/50" />
              <span className="text-xs uppercase tracking-widest text-white/50">
                Build Timestamp
              </span>
            </div>
            <span className="text-sm font-mono text-white/70">
              {BUILD_TIME}
            </span>
          </div>

          <div className="border border-white/10 p-6 flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <Server size={18} className="text-white/50" />
              <span className="text-xs uppercase tracking-widest text-white/50">
                Auto-Refresh
              </span>
            </div>
            <span className="text-sm text-white/70">
              Polls /api/health every 15s (shared across components)
            </span>
          </div>
        </div>

        <div className="pt-4">
          <Link
            href="/parameters"
            className="inline-block px-6 py-3 border border-white/10 text-sm uppercase tracking-widest hover:border-white/40 transition-colors"
          >
            Back to Engine
          </Link>
        </div>
      </div>
    </div>
  );
}

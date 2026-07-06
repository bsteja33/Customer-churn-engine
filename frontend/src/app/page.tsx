"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { Activity, Database, ChevronRight, SquareActivity } from "lucide-react";
import { useFormStore } from "../store/useFormStore";
import { useResultStore } from "../store/useResultStore";
import { useHealthStore } from "../store/useHealthStore";
import { useHealthSubscription } from "../hooks/useHealthSubscription";
import { PRESETS } from "../data/presets";
export default function CommandCenter() {
  const router = useRouter();
  // Subscribe to shared health polling so the home page also benefits
  // from the single shared interval. The rail already subscribes, but
  // explicit subscription here makes the data dependency clear.
  useHealthSubscription();
  const summary = useHealthStore((s) => s.summary);
  const health = useHealthStore((s) => s.health);
  const loadPreset = useFormStore((s) => s.loadPreset);

  const apiConnected = summary === "online" || summary === "degraded";
  const modelLoaded =
    health.phase === "ok" ? health.data.model_loaded : false;

  const handlePreset = (preset: (typeof PRESETS)[number]) => {
    loadPreset({ values: preset.values });
    // Clear any prior results so the next navigation to /analysis is
    // not confused by a stale prediction that no longer matches the form.
    useResultStore.getState().clear();
    router.push("/parameters");
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-72px)] bg-black text-white px-8">
      <div className="max-w-2xl w-full flex flex-col gap-12">
        <div className="flex flex-col gap-4">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tighter">
            Enterprise Churn Engine
          </h1>
          <p className="text-white/60 text-lg md:text-xl font-light tracking-wide max-w-lg">
            Triage, explain, and act on churn risk.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center gap-4 p-6 border border-white/10">
            <Activity className={apiConnected ? "text-white" : "text-white/30"} />
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-widest uppercase">API Engine</span>
              <span className="text-xs text-white/50 tracking-wider">
                {apiConnected ? "Connected" : "Disconnected"}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4 p-6 border border-white/10">
            <Database className={modelLoaded ? "text-white" : "text-white/30"} />
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-widest uppercase">Predictive Model</span>
              <span className="text-xs text-white/50 tracking-wider">
                {modelLoaded ? "Loaded" : "Offline"}
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <span className="text-xs uppercase tracking-widest text-white/40">
            Quickstart
          </span>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {PRESETS.map((preset) => (
              <button
                key={preset.id}
                onClick={() => handlePreset(preset)}
                className="group flex flex-col items-start gap-2 p-5 border border-white/10 hover:border-white/40 transition-colors text-left"
              >
                <div className="flex w-full items-center justify-between">
                  <span className="text-sm font-bold tracking-widest uppercase">
                    {preset.label}
                  </span>
                  <ChevronRight className="text-white/50 group-hover:text-white group-hover:translate-x-1 transition-all" />
                </div>
                <span className="text-xs text-white/50 leading-relaxed">
                  {preset.blurb}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="pt-2">
          <Link
            href="/parameters"
            className="group flex items-center justify-between p-6 border border-white/10 hover:border-white/40 transition-colors duration-500 w-full md:w-2/3"
          >
            <span className="text-sm font-bold tracking-widest uppercase">Enter Engine</span>
            <ChevronRight className="text-white/50 group-hover:text-white group-hover:translate-x-2 transition-all duration-500" />
          </Link>
        </div>

        <div>
          <Link
            href="/status"
            className="group flex items-center justify-between p-4 border border-white/10 hover:border-white/40 transition-colors duration-500 w-full md:w-1/2"
          >
            <div className="flex items-center gap-3">
              <SquareActivity size={16} className="text-white/50 group-hover:text-white transition-colors" />
              <span className="text-xs font-bold tracking-widest uppercase">System Status</span>
            </div>
            <ChevronRight className="text-white/50 group-hover:text-white group-hover:translate-x-2 transition-all duration-500" />
          </Link>
        </div>
      </div>
    </div>
  );
}

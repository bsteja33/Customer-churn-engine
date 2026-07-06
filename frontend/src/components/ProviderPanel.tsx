"use client";

import { useEffect, useState } from "react";
import { X, KeyRound, Cpu, Trash2, CheckCircle2, AlertCircle } from "lucide-react";
import {
  useProviderStore,
  hasKey,
  MODEL_OPTIONS,
  type ModelId,
} from "../store/useProviderStore";
import { fetchModelCatalog } from "../lib/llm";

interface ProviderPanelProps {
  open: boolean;
  onClose: () => void;
}

export function ProviderPanel({ open, onClose }: ProviderPanelProps) {
  const key = useProviderStore((s) => s.key);
  const model = useProviderStore((s) => s.model);
  const setKey = useProviderStore((s) => s.setKey);
  const setModel = useProviderStore((s) => s.setModel);
  const clear = useProviderStore((s) => s.clear);
  // Derived at render time so it can never drift from the
  // persisted ``key`` value.
  const providerHasKey = useProviderStore(hasKey);

  const [draftKey, setDraftKey] = useState(key);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setDraftKey(key);
    setCatalogError(null);
    fetchModelCatalog()
      .then(() => setCatalogLoaded(true))
      .catch((e: unknown) =>
        setCatalogError(e instanceof Error ? e.message : "Catalog unavailable")
      );
    // Re-fetch the catalog only when the panel opens; key edits don't
    // need to re-hit the network.
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!open) return null;

  const handleSave = () => {
    setKey(draftKey);
    onClose();
  };

  const handleClear = () => {
    setDraftKey("");
    clear();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Provider configuration"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md border border-white/15 bg-[#0a0a0a] text-white shadow-2xl"
      >
        <header className="flex items-center justify-between border-b border-white/10 px-5 py-3">
          <div className="flex items-center gap-2">
            <KeyRound size={14} className="text-white/60" />
            <span className="text-xs uppercase tracking-widest text-white/80">
              Provider Configuration
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close panel"
            className="text-white/40 hover:text-white"
          >
            <X size={16} />
          </button>
        </header>

        <div className="flex flex-col gap-5 p-5">
          <p className="text-[11px] text-white/50 leading-relaxed">
            Drop in your LLM provider API key to unlock the
            script-generation endpoint. The key is held in this browser
            only — it never reaches the server unless you make a
            request, and is cleared when you press <em>Clear</em>.
          </p>

          <div className="flex flex-col gap-1.5">
            <label
              htmlFor="provider-key"
              className="text-[10px] uppercase tracking-widest text-white/50"
            >
              API Key
            </label>
            <div className="relative">
              <input
                id="provider-key"
                type="password"
                value={draftKey}
                onChange={(e) => setDraftKey(e.target.value)}
                placeholder="paste key here…"
                autoComplete="off"
                spellCheck={false}
                className="w-full bg-black border border-white/15 px-3 py-2 pr-9 text-xs font-mono text-white placeholder:text-white/30 focus:outline-none focus:border-white/40"
              />
              {providerHasKey && (
                <CheckCircle2
                  size={14}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-green-400"
                  aria-label="Key set"
                />
              )}
            </div>
            {providerHasKey && (
              <span className="text-[10px] text-green-400 font-mono flex items-center gap-1.5">
                <CheckCircle2 size={10} /> Key active for this session
              </span>
            )}
          </div>

          <div className="flex flex-col gap-1.5">
            <label
              htmlFor="provider-model"
              className="text-[10px] uppercase tracking-widest text-white/50 flex items-center gap-1.5"
            >
              <Cpu size={10} /> Inference Model
            </label>
            <select
              id="provider-model"
              value={model}
              onChange={(e) => setModel(e.target.value as ModelId)}
              className="w-full bg-black border border-white/15 px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-white/40"
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label} — {m.apiName}
                </option>
              ))}
            </select>
            {catalogLoaded && (
              <span className="text-[10px] text-white/30 font-mono">
                Catalog synced from backend.
              </span>
            )}
            {catalogError && (
              <span className="text-[10px] text-amber-400 font-mono flex items-center gap-1.5">
                <AlertCircle size={10} /> {catalogError}
              </span>
            )}
          </div>

          <div className="flex items-center justify-between border-t border-white/10 pt-4">
            <button
              type="button"
              onClick={handleClear}
              className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-white/50 hover:text-red-400 transition-colors"
            >
              <Trash2 size={12} /> Clear
            </button>
            <button
              type="button"
              onClick={handleSave}
              className="bg-white text-black px-4 py-1.5 text-[10px] uppercase tracking-widest font-bold hover:bg-white/90 transition-colors"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "../lib/cn";

interface CopyButtonProps {
  value: string;
  label?: string;
  className?: string;
}

/** Clipboard copy with a 1.5s confirmation flash. Falls back to execCommand. */
export function CopyButton({ value, label = "Copy", className }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(id);
  }, [copied]);

  const handleClick = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
      } else {
        const ta = document.createElement("textarea");
        ta.value = value;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-live="polite"
      aria-label={copied ? "Copied" : label}
      className={cn(
        "flex items-center gap-2 px-3 py-1.5 border text-[10px] uppercase tracking-widest transition-colors font-sans",
        copied
          ? "border-green/40 text-green bg-green/5"
          : "border-white/10 text-white/50 hover:text-white hover:border-white/40",
        className
      )}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
      {copied ? "Copied" : label}
    </button>
  );
}

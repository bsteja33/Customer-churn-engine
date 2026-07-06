"use client";

import Link from "next/link";
import { useState } from "react";
import { usePathname } from "next/navigation";
import {
  Activity,
  Database,
  ChevronRight,
  SquareActivity,
  ExternalLink,
  KeyRound,
} from "lucide-react";
import { useHealthStore } from "../store/useHealthStore";
import { useHealthSubscription } from "../hooks/useHealthSubscription";
import { useProviderStore } from "../store/useProviderStore";
import { ProviderPanel } from "./ProviderPanel";
import { cn } from "../lib/cn";

interface RailLink {
  href: string;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
}

const RAIL_LINKS: ReadonlyArray<RailLink> = [
  { href: "/", label: "Engine", icon: Activity },
  { href: "/parameters", label: "Inputs", icon: ChevronRight },
  { href: "/analysis", label: "Results", icon: Database },
  { href: "/status", label: "Status", icon: SquareActivity },
];

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(href + "/");
}

function HealthChip() {
  useHealthSubscription();
  const summary = useHealthStore((s) => s.summary);
  const health = useHealthStore((s) => s.health);

  const dotClass =
    summary === "online"
      ? "bg-green-500 shadow-green-500/60 animate-pulse"
      : summary === "degraded"
        ? "bg-amber-500 shadow-amber-500/60"
        : summary === "offline"
          ? "bg-red-500 shadow-red-500/60"
          : "bg-white/30";

  const tooltip =
    health.phase === "ok"
      ? `Online · ${health.data.latency_ms}ms`
      : health.phase === "error"
        ? `Offline · ${health.message}`
        : "Checking…";

  return (
    <div
      className="flex flex-col items-center gap-1.5"
      role="status"
      aria-live="polite"
      title={tooltip}
    >
      <span
        className={cn(
          "h-2 w-2 rounded-full shadow-[0_0_8px_var(--tw-shadow-color)]",
          dotClass
        )}
        aria-hidden
      />
      <span className="text-[9px] uppercase tracking-widest text-white/40">
        {summary === "online"
          ? "Live"
          : summary === "degraded"
            ? "Warn"
            : summary === "offline"
              ? "Down"
              : "—"}
      </span>
    </div>
  );
}

function ProviderButton({ onClick }: { onClick: () => void }) {
  const providerHasKey = useProviderStore((s) => s.key.trim().length > 0);
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Open provider configuration"
      title={providerHasKey ? "Provider key active" : "Configure provider"}
      className={cn(
        "mt-2 flex h-12 w-12 flex-col items-center justify-center gap-1 transition-colors",
        providerHasKey
          ? "text-green-400 hover:text-green-300"
          : "text-white/40 hover:text-white"
      )}
    >
      <KeyRound size={18} />
      <span className="text-[9px] uppercase tracking-widest">
        {providerHasKey ? "Key" : "API"}
      </span>
    </button>
  );
}

export function LeftRail({ onOpenProvider }: { onOpenProvider: () => void }) {
  const pathname = usePathname() ?? "/";

  return (
    <nav
      aria-label="Primary"
      className="fixed inset-y-0 left-0 z-40 hidden w-[72px] flex-col items-center justify-between border-r border-white/10 bg-black py-6 md:flex"
    >
      <div className="flex flex-col items-center gap-1">
        <div
          className="mb-6 flex h-8 w-8 items-center justify-center border border-white/20 text-[10px] font-bold tracking-widest text-white"
          aria-hidden
        >
          CE
        </div>
        {RAIL_LINKS.map(({ href, label, icon: Icon }) => {
          const active = isActive(pathname, href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              aria-label={label}
              title={label}
              className={cn(
                "group flex h-12 w-12 flex-col items-center justify-center gap-1 transition-colors",
                active
                  ? "bg-white/10 text-white"
                  : "text-white/40 hover:text-white"
              )}
            >
              <Icon size={18} />
              <span className="text-[9px] uppercase tracking-widest">
                {label.slice(0, 4)}
              </span>
            </Link>
          );
        })}
        <a
          href="/docs"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="API docs (opens in new tab)"
          title="API docs"
          className="mt-2 flex h-12 w-12 flex-col items-center justify-center gap-1 text-white/40 transition-colors hover:text-white"
        >
          <ExternalLink size={18} />
          <span className="text-[9px] uppercase tracking-widest">Docs</span>
        </a>
        <ProviderButton onClick={onOpenProvider} />
      </div>
      <HealthChip />
    </nav>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [providerOpen, setProviderOpen] = useState(false);
  return (
    <div className="min-h-screen bg-black text-white">
      <LeftRail onOpenProvider={() => setProviderOpen(true)} />
      <ProviderPanel open={providerOpen} onClose={() => setProviderOpen(false)} />
      <main className="md:pl-[72px]">{children}</main>
    </div>
  );
}

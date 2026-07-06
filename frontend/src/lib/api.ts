/**
 * Centralized API client. Every outbound request passes through
 * ``apiFetch`` so the active provider key + model selection is
 * attached as request headers on every call, regardless of route.
 *
 * Headers:
 *   X-Provider-Key  — the user-supplied LLM provider key, or empty
 *                     if the user has not configured one. The
 *                     backend treats the env-loaded key as the
 *                     fallback when this header is absent, so the
 *                     header-only contract is symmetric and safe.
 *   X-Provider-Model — the model id the user selected in the
 *                     Provider Configuration Panel
 *                     (standard / high_capacity). The backend
 *                     resolves this against its known catalog.
 *
 * Why centralize?
 *   - It eliminates the silent failure mode where a component
 *     imports ``fetch`` directly and bypasses the provider
 *     configuration. Any new API call is forced through this
 *     helper, so the active configuration is observable
 *     everywhere.
 *   - The ``X-Provider-*`` headers become part of the wire
 *     contract: backend logs can attribute each request to a
 *     model, and the per-request key override takes precedence
 *     over the env default.
 */
import { useProviderStore, DEFAULT_MODEL } from "../store/useProviderStore";

export interface ApiFetchOptions extends Omit<RequestInit, "headers"> {
  headers?: HeadersInit;
}

export function apiFetch(
  path: string,
  init: ApiFetchOptions = {},
): Promise<Response> {
  const { key, model } = useProviderStore.getState();
  const headers = new Headers(init.headers);
  if (key.trim()) headers.set("X-Provider-Key", key.trim());
  if (model && model !== DEFAULT_MODEL) {
    headers.set("X-Provider-Model", model);
  }
  return fetch(path, { ...init, headers });
}

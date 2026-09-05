/**
 * Single fetch wrapper: every outbound call goes through here so the
 * provider key and model selection are attached as X-Provider-Key /
 * X-Provider-Model headers consistently.
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

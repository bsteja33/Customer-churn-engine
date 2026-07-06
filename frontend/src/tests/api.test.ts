import { describe, it, expect, beforeEach, vi } from "vitest";
import { apiFetch } from "../lib/api";
import { useProviderStore } from "../store/useProviderStore";

describe("apiFetch (centralized request interceptor)", () => {
  beforeEach(() => {
    localStorage.clear();
    useProviderStore.setState({ key: "", model: "standard" });
    vi.restoreAllMocks();
  });

  it("attaches X-Provider-Key when the user has typed a non-empty key", async () => {
    useProviderStore.setState({ key: "live-test-key", model: "standard" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));
    await apiFetch("/api/predict", { method: "POST" });
    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("X-Provider-Key")).toBe("live-test-key");
    expect(headers.get("X-Provider-Model")).toBeNull();
  });

  it("attaches X-Provider-Model when the user picked a non-default model", async () => {
    useProviderStore.setState({ key: "k", model: "high_capacity" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));
    await apiFetch("/api/llm/models");
    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("X-Provider-Key")).toBe("k");
    expect(headers.get("X-Provider-Model")).toBe("high_capacity");
  });

  it("omits X-Provider-Key when the user has not typed anything", async () => {
    useProviderStore.setState({ key: "", model: "standard" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));
    await apiFetch("/api/health");
    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("X-Provider-Key")).toBeNull();
    expect(headers.get("X-Provider-Model")).toBeNull();
  });

  it("trims whitespace from the key before sending", async () => {
    useProviderStore.setState({ key: "  spaced-key  ", model: "standard" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));
    await apiFetch("/api/predict");
    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("X-Provider-Key")).toBe("spaced-key");
  });

  it("preserves the caller-supplied Content-Type header", async () => {
    useProviderStore.setState({ key: "k", model: "standard" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));
    await apiFetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(headers.get("X-Provider-Key")).toBe("k");
  });

  it("forwards the signal and method to the underlying fetch", async () => {
    useProviderStore.setState({ key: "k", model: "standard" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));
    const controller = new AbortController();
    await apiFetch("/api/predict", {
      method: "POST",
      signal: controller.signal,
    });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/predict");
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(init.signal).toBe(controller.signal);
  });
});

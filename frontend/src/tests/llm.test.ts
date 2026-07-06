import { describe, it, expect, beforeEach, vi } from "vitest";
import { useProviderStore, DEFAULT_MODEL } from "../store/useProviderStore";
import { fetchRetentionScript, fetchModelCatalog } from "../lib/llm";

describe("lib/llm", () => {
  beforeEach(() => {
    localStorage.clear();
    useProviderStore.setState({ key: "", model: DEFAULT_MODEL });
    vi.restoreAllMocks();
  });

  describe("fetchRetentionScript", () => {
    it("POSTs JSON to /api/generate_retention_script and returns the script", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ script: "Offer 20% discount" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await fetchRetentionScript({
        risk_level: "High",
        reasons: "low satisfaction",
      });

      expect(result).toEqual({ script: "Offer 20% discount" });
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe("/api/generate_retention_script");
      expect(init.method).toBe("POST");
      const headers = new Headers(init.headers);
      expect(headers.get("Content-Type")).toBe("application/json");
      expect(JSON.parse(init.body)).toEqual({
        risk_level: "High",
        reasons: "low satisfaction",
      });
    });

    it("injects X-Provider-Key when set in the store", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ script: "ok" }),
      });
      vi.stubGlobal("fetch", fetchMock);
      useProviderStore.setState({ key: "secret-token" });

      await fetchRetentionScript({ risk_level: "Low", reasons: "loyal" });

      const [, init] = fetchMock.mock.calls[0];
      const headers = new Headers(init.headers);
      expect(headers.get("X-Provider-Key")).toBe("secret-token");
    });

    it("injects X-Provider-Model when model differs from default", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ script: "ok" }),
      });
      vi.stubGlobal("fetch", fetchMock);
      useProviderStore.setState({ model: "high_capacity" });

      await fetchRetentionScript({ risk_level: "Medium", reasons: "mid" });

      const [, init] = fetchMock.mock.calls[0];
      const headers = new Headers(init.headers);
      expect(headers.get("X-Provider-Model")).toBe("high_capacity");
    });

    it("omits X-Provider-Key header when store key is empty", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ script: "ok" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      await fetchRetentionScript({ risk_level: "Low", reasons: "ok" });

      const [, init] = fetchMock.mock.calls[0];
      const headers = new Headers(init.headers);
      expect(headers.get("X-Provider-Key")).toBeNull();
    });

    it("throws on non-2xx response", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({ ok: false, status: 500 })
      );
      await expect(
        fetchRetentionScript({ risk_level: "High", reasons: "x" })
      ).rejects.toThrow(/HTTP 500/);
    });
  });

  describe("fetchModelCatalog", () => {
    it("GETs /api/llm/models and returns the catalog", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          models: { standard: "llm-default", high_capacity: "llm-large" },
          default: "llm-default",
        }),
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await fetchModelCatalog();
      expect(result.default).toBe("llm-default");
      expect(result.models.high_capacity).toBe("llm-large");

      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe("/api/llm/models");
      expect(init.method).toBeUndefined();
    });

    it("throws on non-2xx response", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({ ok: false, status: 503 })
      );
      await expect(fetchModelCatalog()).rejects.toThrow(/HTTP 503/);
    });
  });
});

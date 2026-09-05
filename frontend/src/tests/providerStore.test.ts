import { describe, it, expect, beforeEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import {
  useProviderStore,
  MODEL_OPTIONS,
  DEFAULT_MODEL,
  hasKey,
} from "../store/useProviderStore";

const STORAGE_KEY = "churn-provider-config";

describe("useProviderStore", () => {
  beforeEach(() => {
    localStorage.clear();
    useProviderStore.setState({ key: "", model: DEFAULT_MODEL });
  });

  it("starts with empty key and standard model", () => {
    const { result } = renderHook(() => useProviderStore());
    expect(result.current.key).toBe("");
    expect(result.current.model).toBe("standard");
    expect(hasKey(result.current)).toBe(false);
  });

  it("setKey updates key and flips hasKey (derived)", () => {
    const { result } = renderHook(() => useProviderStore());
    act(() => result.current.setKey("abc123"));
    expect(result.current.key).toBe("abc123");
    expect(hasKey(result.current)).toBe(true);
  });

  it("setKey with whitespace-only does not flip hasKey", () => {
    const { result } = renderHook(() => useProviderStore());
    act(() => result.current.setKey("   "));
    expect(hasKey(result.current)).toBe(false);
  });

  it("setModel switches model id", () => {
    const { result } = renderHook(() => useProviderStore());
    act(() => result.current.setModel("high_capacity"));
    expect(result.current.model).toBe("high_capacity");
  });

  it("clear resets everything", () => {
    const { result } = renderHook(() => useProviderStore());
    act(() => {
      result.current.setKey("xyz");
      result.current.setModel("high_capacity");
    });
    act(() => result.current.clear());
    expect(result.current.key).toBe("");
    expect(result.current.model).toBe("standard");
    expect(hasKey(result.current)).toBe(false);
  });
});

describe("useProviderStore persist migration", () => {
  beforeEach(() => {
    localStorage.clear();
    useProviderStore.setState({ key: "", model: DEFAULT_MODEL });
  });

  it("reconciles a legacy v0 payload missing model", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ state: { key: "legacy-key" }, version: 0 })
    );
    await useProviderStore.persist.rehydrate();
    const state = useProviderStore.getState();
    expect(state.key).toBe("legacy-key");
    expect(state.model).toBe(DEFAULT_MODEL);
  });

  it("reconciles a v1 payload preserving both fields", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: { key: "live", model: "high_capacity" },
        version: 1,
      })
    );
    await useProviderStore.persist.rehydrate();
    expect(useProviderStore.getState().key).toBe("live");
    expect(useProviderStore.getState().model).toBe("high_capacity");
  });

  it("drops an unknown model id back to the default", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: { key: "k", model: "gpt-9000" },
        version: 1,
      })
    );
    await useProviderStore.persist.rehydrate();
    expect(useProviderStore.getState().model).toBe(DEFAULT_MODEL);
  });

  it("drops a non-object payload without throwing", async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify("not an object"));
    await expect(useProviderStore.persist.rehydrate()).resolves.not.toThrow();
    expect(useProviderStore.getState().key).toBe("");
    expect(useProviderStore.getState().model).toBe(DEFAULT_MODEL);
  });

  it("strips a stale hasKey field from older builds", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: { key: "k", model: "standard", hasKey: true },
        version: 1,
      })
    );
    await useProviderStore.persist.rehydrate();
    const state = useProviderStore.getState();
    expect(state.key).toBe("k");
    expect(hasKey(state)).toBe(true);
    expect("hasKey" in (state as object)).toBe(false);
  });
});

describe("MODEL_OPTIONS catalog", () => {
  it("exposes a standard and a high_capacity entry", () => {
    const ids = MODEL_OPTIONS.map((m) => m.id);
    expect(ids).toContain("standard");
    expect(ids).toContain("high_capacity");
  });

  it("every option has an id and label", () => {
    for (const m of MODEL_OPTIONS) {
      expect(m.id.length).toBeGreaterThan(0);
      expect(m.label.length).toBeGreaterThan(0);
    }
  });
});

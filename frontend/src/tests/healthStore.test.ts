import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  useHealthStore,
  subscribeHealth,
  __resetHealthForTests,
} from "../store/useHealthStore";

describe("useHealthStore (singleton polling)", () => {
  beforeEach(() => {
    __resetHealthForTests();
  });

  afterEach(() => {
    __resetHealthForTests();
    vi.restoreAllMocks();
  });

  it("starts in the loading phase with unknown summary", () => {
    const state = useHealthStore.getState();
    expect(state.health).toEqual({ phase: "loading" });
    expect(state.summary).toBe("unknown");
  });

  it("subscribeHealth returns an unsubscribe function", () => {
    const unsub = subscribeHealth();
    expect(typeof unsub).toBe("function");
    unsub();
  });

  it("summary is online when status is healthy and model is loaded", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            status: "healthy",
            model_loaded: true,
            model_path: "/tmp/model.cbm",
          }),
          { status: 200 }
        )
      );
    subscribeHealth();
    // Let the immediate poll complete.
    await new Promise((r) => setTimeout(r, 10));
    const state = useHealthStore.getState();
    expect(state.health.phase).toBe("ok");
    expect(state.summary).toBe("online");
    fetchMock.mockRestore();
  });

  it("summary is degraded when model_loaded is false", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "healthy",
          model_loaded: false,
          model_path: "/tmp/model.cbm",
        }),
        { status: 200 }
      )
    );
    subscribeHealth();
    await new Promise((r) => setTimeout(r, 10));
    expect(useHealthStore.getState().summary).toBe("degraded");
    fetchMock.mockRestore();
  });

  it("summary is offline when the fetch rejects", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockRejectedValue(new Error("network down"));
    subscribeHealth();
    await new Promise((r) => setTimeout(r, 10));
    const state = useHealthStore.getState();
    expect(state.health.phase).toBe("error");
    expect(state.summary).toBe("offline");
    expect((state.health as { message: string }).message).toBe("network down");
    fetchMock.mockRestore();
  });

  it("summary is offline when the server returns a non-2xx status", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("nope", { status: 503 }));
    subscribeHealth();
    await new Promise((r) => setTimeout(r, 10));
    const state = useHealthStore.getState();
    expect(state.health.phase).toBe("error");
    expect(state.summary).toBe("offline");
    expect((state.health as { message: string }).message).toBe("HTTP 503");
    fetchMock.mockRestore();
  });

  it("refresh() forces an immediate poll without affecting subscribers", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            status: "healthy",
            model_loaded: true,
            model_path: "/tmp/model.cbm",
          }),
          { status: 200 }
        )
      );
    subscribeHealth();
    // Drain the immediate call.
    await new Promise((r) => setTimeout(r, 10));
    const callsAfterSubscribe = fetchMock.mock.calls.length;
    useHealthStore.getState().refresh();
    await new Promise((r) => setTimeout(r, 10));
    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsAfterSubscribe);
    fetchMock.mockRestore();
  });
});

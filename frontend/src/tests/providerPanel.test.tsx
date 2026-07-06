import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ProviderPanel } from "../components/ProviderPanel";
import { useProviderStore, hasKey } from "../store/useProviderStore";

vi.mock("../lib/llm", () => ({
  fetchModelCatalog: vi.fn().mockResolvedValue({
    models: { standard: "llm-default", high_capacity: "llm-large" },
    default: "llm-default",
  }),
}));

describe("ProviderPanel", () => {
  beforeEach(() => {
    localStorage.clear();
    useProviderStore.setState({ key: "", model: "standard" });
  });

  it("renders nothing when closed", () => {
    const { container } = render(
      <ProviderPanel open={false} onClose={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders the API key input and model select when open", async () => {
    render(<ProviderPanel open={true} onClose={() => {}} />);
    expect(screen.getByLabelText("API Key")).toBeTruthy();
    expect(screen.getByLabelText("Inference Model")).toBeTruthy();
    await waitFor(() => {
      expect(screen.queryByText(/Catalog synced/i)).toBeTruthy();
    });
  });

  it("Save button writes the draft key to the store", () => {
    render(<ProviderPanel open={true} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText("API Key"), {
      target: { value: "live-key-123" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Save/i }));
    const state = useProviderStore.getState();
    expect(state.key).toBe("live-key-123");
    expect(hasKey(state)).toBe(true);
  });

  it("Clear button resets the key", () => {
    useProviderStore.setState({ key: "preexisting" });
    render(<ProviderPanel open={true} onClose={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /Clear/i }));
    const state = useProviderStore.getState();
    expect(state.key).toBe("");
    expect(hasKey(state)).toBe(false);
  });

  it("model selection writes to the store", () => {
    render(<ProviderPanel open={true} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText("Inference Model"), {
      target: { value: "high_capacity" },
    });
    expect(useProviderStore.getState().model).toBe("high_capacity");
  });

  it("Close button dismisses the panel", () => {
    const onClose = vi.fn();
    render(<ProviderPanel open={true} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: /Close panel/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});

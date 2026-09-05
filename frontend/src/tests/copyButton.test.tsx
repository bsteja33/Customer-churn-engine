import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CopyButton } from "../components/CopyButton";

describe("CopyButton", () => {
  const originalClipboard = navigator.clipboard;

  beforeEach(() => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  afterEach(() => {
    Object.assign(navigator, { clipboard: originalClipboard });
    vi.restoreAllMocks();
  });

  it("renders the default label", () => {
    render(<CopyButton value="hello" />);
    expect(screen.getByRole("button", { name: /copy/i })).not.toBeNull();
  });

  it("writes the value to the clipboard and flips to 'Copied'", async () => {
    render(<CopyButton value="retention text" />);
    const button = screen.getByRole("button", { name: /copy/i });
    fireEvent.click(button);
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("retention text");
    // After click, the button label changes to "Copied".
    expect(await screen.findByRole("button", { name: /copied/i })).not.toBeNull();
  });

  it("does not throw when clipboard API is unavailable", async () => {
    Object.assign(navigator, { clipboard: undefined });
    // jsdom doesn't expose document.execCommand by default; stub it.
    const w = window as unknown as { execCommand?: (cmd: string) => boolean };
    const original = w.execCommand;
    w.execCommand = () => true;
    try {
      render(<CopyButton value="fallback" />);
      const button = screen.getByRole("button", { name: /copy/i });
      expect(() => fireEvent.click(button)).not.toThrow();
    } finally {
      if (original) {
        w.execCommand = original;
      } else {
        w.execCommand = undefined;
      }
    }
  });
});

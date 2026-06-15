import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AnimatedPrice } from "./AnimatedPrice";

// NumberFlow renders a custom element with per-digit nodes for its roll animation, which doesn't
// upgrade under jsdom. Mock it to a plain text node so these tests assert behaviour, not its DOM.
vi.mock("@number-flow/react", () => ({
  default: ({ value }: { value: number }) => String(value),
}));

describe("AnimatedPrice", () => {
  it("renders the numeric value", () => {
    render(<AnimatedPrice value={295.4} decimals={2} />);
    expect(screen.getByText("295.4")).toBeInTheDocument();
  });

  it("renders an em dash for NaN (no NumberFlow)", () => {
    render(<AnimatedPrice value={NaN} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("updates to the new value without throwing when it changes", () => {
    // jsdom has no Element.animate, so the flash path must be a guarded no-op.
    const { rerender } = render(<AnimatedPrice value={100} />);
    rerender(<AnimatedPrice value={101.5} />);
    expect(screen.getByText("101.5")).toBeInTheDocument();
  });
});

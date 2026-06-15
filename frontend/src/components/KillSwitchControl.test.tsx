import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { KillSwitchControl } from "./KillSwitchControl";

describe("KillSwitchControl", () => {
  it("renders the armed state and an Engage action when not engaged", () => {
    render(<KillSwitchControl engaged={false} busy={false} onToggle={vi.fn()} />);
    expect(screen.getByText("TRADING ARMED")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /engage kill switch/i })).toBeEnabled();
  });

  it("renders the halted state and a Release action when engaged", () => {
    render(<KillSwitchControl engaged busy={false} onToggle={vi.fn()} />);
    expect(screen.getByText("ORDERS HALTED")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /release kill switch/i })).toBeInTheDocument();
  });

  it("toggles to the opposite state on click", async () => {
    const onToggle = vi.fn();
    const user = userEvent.setup();
    render(<KillSwitchControl engaged={false} busy={false} onToggle={onToggle} />);

    await user.click(screen.getByRole("button", { name: /engage kill switch/i }));
    expect(onToggle).toHaveBeenCalledExactlyOnceWith(true);
  });

  it("disables the button while busy", () => {
    render(<KillSwitchControl engaged={false} busy onToggle={vi.fn()} />);
    expect(screen.getByRole("button", { name: /engage kill switch/i })).toBeDisabled();
  });
});

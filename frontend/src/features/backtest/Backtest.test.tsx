import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../lib/api";
import type { BacktestResult } from "../../lib/types";
import { Backtest } from "./Backtest";

// EquityCurve renders a lightweight-charts canvas that doesn't upgrade under jsdom, and DataGrid is
// AG Grid (heavy, CSS-dependent). Mock both to simple inspectable nodes so these tests assert the
// screen's behaviour, not the widgets'.
vi.mock("../../components/EquityCurve", () => ({
  EquityCurve: ({ data, benchmark }: { data: unknown[]; benchmark?: unknown[] }) => (
    <div data-testid="equity-curve" data-points={data.length} data-benchmark={benchmark?.length ?? 0} />
  ),
}));

vi.mock("../../components/DataGrid", () => ({
  DataGrid: ({ rows }: { rows: unknown[] }) => <div data-testid="trades-grid" data-rows={rows.length} />,
}));

// Hoisted so the (also-hoisted) vi.mock factory below can safely reference it.
const { backtest } = vi.hoisted(() => ({ backtest: vi.fn<() => Promise<BacktestResult>>() }));
vi.mock("../../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../../lib/api")>("../../lib/api");
  return { ...actual, api: { ...actual.api, backtest: () => backtest() } };
});

const FULL_RESULT: BacktestResult = {
  starting_equity: 100000,
  ending_equity: 112500,
  total_return_pct: 12.5,
  max_drawdown_pct: 4.2,
  sharpe: 1.8,
  sortino: 2.4,
  annualized_return_pct: 9.1,
  volatility: 11.3,
  win_rate: 60,
  profit_factor: 1.9,
  avg_win: 250,
  avg_loss: -120,
  num_trades: 8,
  decision_count: 20,
  benchmark_total_return_pct: 7.4,
  equity_curve: [
    { t: "2026-01-01T00:00:00Z", equity: 100000 },
    { t: "2026-01-02T00:00:00Z", equity: 112500 },
  ],
  benchmark_curve: [
    { t: "2026-01-01T00:00:00Z", equity: 100000 },
    { t: "2026-01-02T00:00:00Z", equity: 107400 },
  ],
  trades: [{ symbol: "AAPL", side: "buy", quantity: 10, price: 190.25, filled_at: "2026-01-01T15:00:00Z" }],
};

beforeEach(() => {
  backtest.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("Backtest", () => {
  it("shows a prompt empty state before a backtest is run", () => {
    render(<Backtest refreshKey={0} />);
    expect(screen.getByRole("button", { name: /run backtest/i })).toBeEnabled();
    expect(screen.getByText(/run a backtest to see the equity curve/i)).toBeInTheDocument();
  });

  it("renders metrics, curve (with benchmark), and trades after a successful run", async () => {
    backtest.mockResolvedValue(FULL_RESULT);
    const user = userEvent.setup();
    render(<Backtest refreshKey={0} />);

    await user.click(screen.getByRole("button", { name: /run backtest/i }));

    await waitFor(() => expect(screen.getByTestId("equity-curve")).toBeInTheDocument());
    const curve = screen.getByTestId("equity-curve");
    expect(curve).toHaveAttribute("data-points", "2");
    expect(curve).toHaveAttribute("data-benchmark", "2");

    // Headline metrics render.
    expect(screen.getByText("12.5%")).toBeInTheDocument(); // total return
    expect(screen.getByText("7.4%")).toBeInTheDocument(); // benchmark return
    expect(screen.getByText("1.8")).toBeInTheDocument(); // sharpe
    expect(screen.getByText("60%")).toBeInTheDocument(); // win rate

    expect(screen.getByTestId("trades-grid")).toHaveAttribute("data-rows", "1");
  });

  it("coerces string numerics and renders em-dash for nulls", async () => {
    backtest.mockResolvedValue({
      ...FULL_RESULT,
      total_return_pct: "3.5", // string numeric, as the backend's Decimal serialization sends
      sharpe: null,
      sortino: null,
      win_rate: null,
      benchmark_total_return_pct: null,
    });
    const user = userEvent.setup();
    render(<Backtest refreshKey={0} />);
    await user.click(screen.getByRole("button", { name: /run backtest/i }));

    await waitFor(() => expect(screen.getByText("3.5%")).toBeInTheDocument());
    // Sharpe + Sortino are null -> "—" appears at least twice.
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
  });

  it("shows an explanatory empty state on a 409 (not configured)", async () => {
    backtest.mockRejectedValue(new ApiError(409, "backtesting is not configured"));
    const user = userEvent.setup();
    render(<Backtest refreshKey={0} />);
    await user.click(screen.getByRole("button", { name: /run backtest/i }));

    await waitFor(() =>
      expect(screen.getByText(/backtesting is not configured/i)).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("equity-curve")).not.toBeInTheDocument();
  });

  it("surfaces a non-409 error as an error message", async () => {
    backtest.mockRejectedValue(new ApiError(500, "boom"));
    const user = userEvent.setup();
    render(<Backtest refreshKey={0} />);
    await user.click(screen.getByRole("button", { name: /run backtest/i }));

    await waitFor(() => expect(screen.getByText(/boom/)).toBeInTheDocument());
    expect(screen.queryByTestId("equity-curve")).not.toBeInTheDocument();
  });

  it("renders headline cards section once a result exists", async () => {
    backtest.mockResolvedValue(FULL_RESULT);
    const user = userEvent.setup();
    render(<Backtest refreshKey={0} />);
    await user.click(screen.getByRole("button", { name: /run backtest/i }));

    await waitFor(() => expect(screen.getByText("Profit factor")).toBeInTheDocument());
    // The trade count surfaces in the Trades card heading.
    expect(screen.getByText(/Trades \(1\)/i)).toBeInTheDocument();
  });
});

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../lib/api";
import type { OrderCancelResult, StoredEventDto } from "../../lib/types";
import { Blotter } from "./Blotter";

// AG Grid is heavy and CSS-dependent under jsdom; mock DataGrid to a simple table that renders
// each column's cell (including the Cancel action cell renderer) so we can drive the cancel flow.
vi.mock("../../components/DataGrid", () => {
  return {
    DataGrid: <T,>({
      rows,
      columns,
    }: {
      rows: T[];
      // Minimal shape we rely on from ag-grid's ColDef in these tests.
      columns: {
        headerName?: string;
        field?: keyof T;
        cellRenderer?: (params: { data: T }) => unknown;
        valueFormatter?: (params: { data: T; value: unknown }) => string;
      }[];
    }) => (
      <table>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((col, j) => {
                let content: unknown = col.field ? row[col.field] : null;
                if (col.valueFormatter) content = col.valueFormatter({ data: row, value: content });
                if (col.cellRenderer) content = col.cellRenderer({ data: row });
                return <td key={j}>{content as ReactNode}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    ),
  };
});

// Hoisted so the (also-hoisted) vi.mock factory below can safely reference them.
const { events, cancelOrder } = vi.hoisted(() => ({
  events: vi.fn<() => Promise<StoredEventDto[]>>(),
  cancelOrder: vi.fn<(id: string) => Promise<OrderCancelResult>>(),
}));
vi.mock("../../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../../lib/api")>("../../lib/api");
  return {
    ...actual,
    api: { ...actual.api, events: () => events(), cancelOrder: (id: string) => cancelOrder(id) },
  };
});

let seq = 0;
function evt(event_type: string, payload: Record<string, unknown>): StoredEventDto {
  seq += 1;
  return {
    event_id: `e${seq}`,
    stream_id: "s",
    seq,
    event_type,
    schema_version: 1,
    occurred_at: "2026-01-01T00:00:00.000Z",
    recorded_at: "2026-01-01T00:00:00.000Z",
    correlation_id: "c",
    payload,
  };
}

// An open order (submitted, not terminal) and a filled (terminal) order.
const OPEN_ORDER = [
  evt("order.intent", {
    client_order_id: "open-1",
    sleeve_id: "a",
    symbol: "AAPL",
    side: "buy",
    quantity: "10",
  }),
  evt("order.submitted", { client_order_id: "open-1", broker_order_id: "b-1" }),
];
const FILLED_ORDER = [
  evt("order.intent", {
    client_order_id: "done-1",
    sleeve_id: "a",
    symbol: "MSFT",
    side: "buy",
    quantity: "5",
  }),
  evt("order.status_changed", { client_order_id: "done-1", status: "filled" }),
];

beforeEach(() => {
  events.mockReset();
  cancelOrder.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("Blotter cancel action", () => {
  it("shows a Cancel button only on open orders", async () => {
    events.mockResolvedValue([...OPEN_ORDER, ...FILLED_ORDER]);
    render(<Blotter refreshKey={0} />);

    await waitFor(() => expect(screen.getByText("AAPL")).toBeInTheDocument());
    const cancelButtons = screen.getAllByRole("button", { name: /^cancel$/i });
    expect(cancelButtons).toHaveLength(1);
  });

  it("calls cancelOrder with the client_order_id and shows optimistic state", async () => {
    events.mockResolvedValue([...OPEN_ORDER]);
    cancelOrder.mockResolvedValue({ client_order_id: "open-1", status: "canceled" });
    const user = userEvent.setup();
    render(<Blotter refreshKey={0} />);

    await waitFor(() => expect(screen.getByText("AAPL")).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(cancelOrder).toHaveBeenCalledExactlyOnceWith("open-1");
    // Optimistic: the status cell flips to a canceling label (the button also reads "Canceling…").
    await waitFor(() => expect(screen.getByText("canceling…")).toBeInTheDocument());
  });

  it("surfaces an error and clears the in-flight flag when the cancel fails", async () => {
    events.mockResolvedValue([...OPEN_ORDER]);
    cancelOrder.mockRejectedValue(new ApiError(409, "order already filled"));
    const user = userEvent.setup();
    render(<Blotter refreshKey={0} />);

    await waitFor(() => expect(screen.getByText("AAPL")).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /^cancel$/i }));

    await waitFor(() => expect(screen.getByText(/order already filled/i)).toBeInTheDocument());
    // The Cancel button is enabled again after the failure.
    expect(screen.getByRole("button", { name: /^cancel$/i })).toBeEnabled();
  });
});

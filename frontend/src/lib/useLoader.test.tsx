import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { useLoader } from "./useLoader";

// A tiny probe component that surfaces the hook's state into the DOM so we can assert the
// loading -> data and loading -> error transitions through the real React lifecycle. The
// loader is built fresh each render (mirroring real call sites like `() => api.signalFeed(...)`)
// so it is `deps` — not function identity — that decides when the hook re-runs.
function Probe({ fetcher, dep }: { fetcher: () => Promise<string>; dep: number }) {
  const { data, loading, error, reload } = useLoader(() => fetcher(), [dep]);
  return (
    <div>
      <span data-testid="loading">{loading ? "loading" : "idle"}</span>
      <span data-testid="data">{data ?? ""}</span>
      <span data-testid="error">{error ?? ""}</span>
      <button onClick={reload}>reload</button>
    </div>
  );
}

describe("useLoader", () => {
  it("transitions from loading to resolved data", async () => {
    const loader = vi.fn(() => Promise.resolve("hello"));
    render(<Probe fetcher={loader} dep={1} />);

    // Initial render is in the loading state.
    expect(screen.getByTestId("loading")).toHaveTextContent("loading");

    await waitFor(() => expect(screen.getByTestId("data")).toHaveTextContent("hello"));
    expect(screen.getByTestId("loading")).toHaveTextContent("idle");
    expect(screen.getByTestId("error")).toHaveTextContent("");
    expect(loader).toHaveBeenCalledTimes(1);
  });

  it("captures a rejected loader's message in error and clears loading", async () => {
    const loader = vi.fn(() => Promise.reject(new Error("kaboom")));
    render(<Probe fetcher={loader} dep={1} />);

    await waitFor(() => expect(screen.getByTestId("error")).toHaveTextContent("kaboom"));
    expect(screen.getByTestId("loading")).toHaveTextContent("idle");
    expect(screen.getByTestId("data")).toHaveTextContent("");
  });

  it("re-runs the loader when deps change", async () => {
    let value = "first";
    const loader = vi.fn(() => Promise.resolve(value));
    const { rerender } = render(<Probe fetcher={loader} dep={1} />);
    await waitFor(() => expect(screen.getByTestId("data")).toHaveTextContent("first"));

    value = "second";
    rerender(<Probe fetcher={loader} dep={2} />);
    await waitFor(() => expect(screen.getByTestId("data")).toHaveTextContent("second"));
    expect(loader).toHaveBeenCalledTimes(2);
  });

  it("refetches data on reload()", async () => {
    let value = "v1";
    const loader = vi.fn(() => Promise.resolve(value));
    render(<Probe fetcher={loader} dep={1} />);
    await waitFor(() => expect(screen.getByTestId("data")).toHaveTextContent("v1"));

    value = "v2";
    await userEvent.click(screen.getByText("reload"));
    await waitFor(() => expect(screen.getByTestId("data")).toHaveTextContent("v2"));
    expect(loader).toHaveBeenCalledTimes(2);
  });
});

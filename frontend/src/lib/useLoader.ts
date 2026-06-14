import { useCallback, useEffect, useState } from "react";

interface LoaderState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

// Runs an async loader on mount and whenever `deps` change, exposing data/loading/error.
export function useLoader<T>(loader: () => Promise<T>, deps: readonly unknown[]): LoaderState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // The loader closes over `deps`; intentionally re-created when they change.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(loader, deps);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    run()
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [run]);

  const reload = useCallback(() => {
    void run().then(setData);
  }, [run]);

  return { data, loading, error, reload };
}

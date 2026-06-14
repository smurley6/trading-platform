// Streaming real-time view via FINOS Perspective (FR-UI-4): the live conviction feed and
// streaming P&L. Data is pushed in on every websocket tick; Perspective re-renders.

import "@finos/perspective-viewer";
import "@finos/perspective-viewer-datagrid";
import "@finos/perspective-viewer/dist/css/themes.css";
import { useEffect, useRef } from "react";
import { perspectiveClient } from "../lib/perspective";

// Minimal structural types for the bits of the Perspective API we touch — kept local so a
// patch-level type change upstream doesn't break our build, and so the viewer's `any`-typed
// methods don't leak into our code.
interface PerspectiveTable {
  replace(rows: Record<string, unknown>[]): Promise<void>;
  delete(): Promise<void>;
}
interface PerspectiveWorker {
  table(data: Record<string, unknown>[]): Promise<PerspectiveTable>;
}
interface PerspectiveViewer extends HTMLElement {
  load(table: PerspectiveTable): Promise<void>;
  restore(config: Record<string, unknown>): Promise<void>;
}

interface StreamingTableProps {
  data: Record<string, unknown>[];
  config?: Record<string, unknown>;
  height?: number;
}

export function StreamingTable({ data, config, height = 320 }: StreamingTableProps) {
  const viewerRef = useRef<PerspectiveViewer | null>(null);
  const tableRef = useRef<PerspectiveTable | null>(null);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || data.length === 0) return;
    let disposed = false;

    void (async () => {
      try {
        const worker = (await perspectiveClient()) as unknown as PerspectiveWorker;
        if (disposed) return;
        if (!tableRef.current) {
          tableRef.current = await worker.table(data);
          await viewer.load(tableRef.current);
          if (config) await viewer.restore(config);
        } else {
          await tableRef.current.replace(data);
        }
      } catch (error) {
        console.error("perspective render failed", error);
      }
    })();

    return () => {
      disposed = true;
    };
  }, [data, config]);

  useEffect(() => {
    return () => {
      void tableRef.current?.delete();
      tableRef.current = null;
    };
  }, []);

  return (
    <perspective-viewer
      ref={(element) => {
        viewerRef.current = element;
      }}
      style={{ height, width: "100%" }}
    />
  );
}

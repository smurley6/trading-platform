// Perspective v3 bootstrap. The engine and viewer each load a WASM module; importing them
// with `?url` makes Vite emit the .wasm files as assets and hands us their served URLs, so
// the streaming views work from a production build (not just dev).

import { init_server, worker } from "@finos/perspective";
import { init_client } from "@finos/perspective-viewer";
import SERVER_WASM from "@finos/perspective/dist/wasm/perspective-server.wasm?url";
import VIEWER_WASM from "@finos/perspective-viewer/dist/wasm/perspective-viewer.wasm?url";

type Client = Awaited<ReturnType<typeof worker>>;

let clientPromise: Promise<Client> | null = null;

/** Initialize Perspective once and return a shared client. */
export function perspectiveClient(): Promise<Client> {
  clientPromise ??= (async () => {
    init_server(fetch(SERVER_WASM));
    await init_client(fetch(VIEWER_WASM));
    return worker();
  })();
  return clientPromise;
}

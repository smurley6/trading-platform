/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The backend runs on :8000. We proxy /api and /ws to it so the SPA stays same-origin
// (no CORS, no hardcoded host). The REST client talks to "/api/*"; the WebSocket client
// connects to "/ws/events".
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/ws": {
        target: BACKEND.replace(/^http/, "ws"),
        ws: true,
      },
    },
  },
  // Perspective ships WASM and a web-worker; excluding it from dep pre-bundling lets Vite
  // serve those assets correctly.
  optimizeDeps: {
    exclude: [
      "@finos/perspective",
      "@finos/perspective-viewer",
      "@finos/perspective-viewer-datagrid",
    ],
    // The Perspective packages are excluded above, so Vite never crawls into them to
    // discover their CommonJS transitive deps. List those here so they still get the
    // CJS->ESM interop treatment in dev (chroma-js exports no ESM `default` otherwise).
    include: ["chroma-js"],
  },
  worker: { format: "es" },
  build: { target: "es2021", chunkSizeWarningLimit: 2000 },
  // Vitest configuration (jsdom DOM, jest-dom matchers via setup, global test APIs).
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
});

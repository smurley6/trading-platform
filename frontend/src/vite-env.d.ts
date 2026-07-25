/// <reference types="vite/client" />

// The <perspective-viewer> custom element's JSX typing is provided by
// "@finos/perspective-viewer" itself, so no manual declaration is needed here.

// Build-time config. VITE_WS_URL points the live WebSocket streams (/ws/events,
// /ws/quotes) at the backend origin in production, where the static host can't proxy
// WS upgrades. Unset in dev (the Vite proxy handles /ws/*). See lib/wsBase.ts.
// VITE_API_KEY is the backend's `API_KEY` (sent as `X-API-Key` on every REST call);
// unset means the backend is in its default-open mode. See lib/apiKey.ts.
interface ImportMetaEnv {
  readonly VITE_WS_URL?: string;
  readonly VITE_API_URL?: string;
  readonly VITE_API_KEY?: string;
}

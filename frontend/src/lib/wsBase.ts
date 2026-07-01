// Resolves the WebSocket origin for the live streams (/ws/events, /ws/quotes).
//
// In production the SPA is served as static files (e.g. a CDN / static host) that cannot
// proxy WebSocket upgrades, so the sockets must point directly at the backend. Set
// VITE_WS_URL at build time to the backend's WS origin (e.g. "wss://api.example.com").
// When it is unset (local dev), we fall back to the current page origin — where the Vite
// dev server proxies /ws/* to the backend — preserving today's behaviour exactly.
export function wsBase(): string {
  const configured = import.meta.env.VITE_WS_URL?.trim();
  if (configured) return configured.replace(/\/+$/, "");
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}`;
}

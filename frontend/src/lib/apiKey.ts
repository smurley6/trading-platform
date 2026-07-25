// Resolves the API key the backend's `ApiKeyAuth` guard expects (backend `interface/auth.py`).
//
// The backend is default-open: with no `API_KEY` configured every route is unguarded. Once a key
// IS configured it protects all writes plus the sensitive reads — `GET /events`, `GET /capital`,
// `GET /positions/live` — and the `/ws/events` firehose. The guard accepts either
// `Authorization: Bearer <key>` or `X-API-Key: <key>`; we use `X-API-Key` (the "simple clients"
// header the backend documents) so we never collide with a future bearer-token scheme.
//
// Supplied at build time as `VITE_API_KEY`, matching how `VITE_API_URL`/`VITE_WS_URL` already
// configure the deployed SPA. Unset (today's dev + deployed config) means "no key": every request
// goes out exactly as it does now, with no header added at all — never a header whose value is
// `undefined`. Read inside the function (not captured in a module const) so the resolution is
// observable in tests, same as `wsBase()`.
export function apiKey(): string | null {
  const configured = import.meta.env.VITE_API_KEY?.trim();
  return configured ? configured : null;
}

/** The auth header pair for the configured key, or an empty object when no key is set. Spread
 *  into a request's headers so the absent case adds nothing. */
export function authHeaders(): Record<string, string> {
  const key = apiKey();
  return key === null ? {} : { "X-API-Key": key };
}

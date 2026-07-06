// UI Phase A — injects the §3 design tokens as CSS custom properties on :root, layered onto the
// Salt dark theme (SaltProvider mode="dark" stays wired in main.tsx; this overlays the operator
// console palette on top of it so Salt-rendered chrome and our own components share one set of
// CSS vars), and registers the IBM Plex font faces with `font-display: optional` (§3.5 — no CLS
// on font swap). Call once at startup, before first paint, from main.tsx.
//
// CSS var values are generated FROM tokens.ts (`cssVars`) rather than duplicated here, so
// tokens.ts stays the single source of truth per docs/UI-DESIGN.md §3.
//
// Fonts are registered via the FontFace API (rather than fontsource's own CSS, which ships
// `font-display: swap`) so we can set `font-display: optional` per §3.5 while still getting
// the woff2 files served as fingerprinted assets through Vite's `?url` asset pipeline — same
// pattern as lib/perspective.ts's WASM imports. Only latin, weights 400/500/600 (§3.2).

import ibmPlexMono400 from "@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2?url";
import ibmPlexMono500 from "@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2?url";
import ibmPlexMono600 from "@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-600-normal.woff2?url";
import ibmPlexSans400 from "@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2?url";
import ibmPlexSans500 from "@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-500-normal.woff2?url";
import ibmPlexSans600 from "@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2?url";
import { cssVars } from "./tokens";

/** Apply every §3 CSS custom property to `document.documentElement`. Idempotent. */
export function applyThemeVars(): void {
  const root = document.documentElement.style;
  for (const [name, value] of Object.entries(cssVars)) {
    root.setProperty(name, value);
  }
}

// 2026-07-06 site-wide font unification: the 600 face is registered for the weight RANGE
// "600 800" (FontFace accepts ranges), not just "600". Legacy `text.display/h1/h2/h3`,
// `pillStyle`, and other 700/800-weight requests now resolve to the real Plex 600 glyphs
// instead of the browser's synthesized (faux-bold) 600 face — still only three genuine
// weights shipped (§3.2: 400/500/600 only), just mapped onto the full weight range legacy
// screens still request.
const FONT_FACES: { family: string; url: string; weight: string }[] = [
  { family: "IBM Plex Sans", url: ibmPlexSans400, weight: "400" },
  { family: "IBM Plex Sans", url: ibmPlexSans500, weight: "500" },
  { family: "IBM Plex Sans", url: ibmPlexSans600, weight: "600 800" },
  { family: "IBM Plex Mono", url: ibmPlexMono400, weight: "400" },
  { family: "IBM Plex Mono", url: ibmPlexMono500, weight: "500" },
  { family: "IBM Plex Mono", url: ibmPlexMono600, weight: "600 800" },
];

let fontsRegistered = false;

/** Register the IBM Plex latin subset (400/500/600) with `font-display: optional`. Idempotent. */
export function registerFonts(): void {
  if (fontsRegistered) return;
  fontsRegistered = true;
  if (typeof FontFace === "undefined") return; // jsdom in tests has no FontFace API
  for (const { family, url, weight } of FONT_FACES) {
    const face = new FontFace(family, `url(${url}) format("woff2")`, {
      weight,
      display: "optional",
    });
    document.fonts.add(face);
    void face.load().catch(() => {
      // A failed font fetch (offline, blocked CDN — though these are self-hosted so this
      // should be rare) should never break the app; the fallback stack in tokens.ts covers it.
    });
  }
}

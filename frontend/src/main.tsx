import { SaltProvider } from "@salt-ds/core";
import "@salt-ds/theme/index.css";
import React from "react";
import ReactDOM from "react-dom/client";
import { Toaster } from "sonner";
import { App } from "./App";
import { ToastProvider } from "./components/Toast";
import "./index.css";
import { applyThemeVars, registerFonts } from "./lib/theme";
import { type as typeTokens } from "./lib/tokens";
import "./theme.css";

// UI Phase A: apply the §3 CSS custom properties + register IBM Plex (font-display: optional,
// §3.5) before first paint, so there is no flash of unstyled/un-tokened content.
applyThemeVars();
registerFonts();

const container = document.getElementById("root");
if (!container) {
  throw new Error("missing #root element");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <SaltProvider mode="dark">
      <ToastProvider>
        <App />
        {/* §7 toasts: bottom-right, mono, 4s (sonner default), failures persist (set per-call in
            components/Toast.tsx via duration: Infinity on .error()). */}
        <Toaster
          position="bottom-right"
          duration={4000}
          toastOptions={{ style: { fontFamily: "var(--font-mono)", fontSize: typeTokens.scale.default } }}
        />
      </ToastProvider>
    </SaltProvider>
  </React.StrictMode>,
);

import { SaltProvider } from "@salt-ds/core";
import "@salt-ds/theme/index.css";
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { ToastProvider } from "./components/Toast";
import "./index.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("missing #root element");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <SaltProvider mode="dark">
      <ToastProvider>
        <App />
      </ToastProvider>
    </SaltProvider>
  </React.StrictMode>,
);

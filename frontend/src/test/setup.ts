// Vitest global setup: extends `expect` with jest-dom matchers (e.g. toBeInTheDocument,
// toHaveTextContent) and clears the DOM/mocks between tests so suites stay isolated.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});

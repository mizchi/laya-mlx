import { expect, type Page, test as base } from "@playwright/test";

export const text = (page: Page, id: string) => page.locator(`#${id}`).textContent();

/**
 * Collects console errors and uncaught page errors for the duration of a test. Most tests in
 * this suite assert this is empty except a bad-model-URL test, which expects at least one (it
 * hits the `start().catch` handler's `console.error(error)` in main.ts).
 */
// onnxruntime-web itself (not this app) logs this informational EP-assignment note through
// console.error when a graph mixes execution providers; it is expected on every real-model run
// and is not a bug, so it is filtered out rather than muting console-error checks altogether.
const BENIGN_CONSOLE_ERROR = /VerifyEachNodeIsAssignedToAnEp/;

export const test = base.extend<{ pageErrors: string[] }>({
  pageErrors: async ({ page }, use) => {
    const errors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error" && !BENIGN_CONSOLE_ERROR.test(message.text())) {
        errors.push(message.text());
      }
    });
    page.on("pageerror", (error) => errors.push(String(error)));
    await use(errors);
  },
});

export { expect };

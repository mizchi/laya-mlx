import { expect, type Page, test as base } from "@playwright/test";

const text = (page: Page, id: string) => page.locator(`#${id}`).textContent();

/**
 * Collects console errors and uncaught page errors for the duration of a test. Every test in
 * this file asserts this is empty except the bad-model-URL test, which expects one (it hits the
 * `start().catch` handler's `console.error(error)` in main.ts).
 */
// onnxruntime-web itself (not this app) logs this informational EP-assignment note through
// console.error when a graph mixes execution providers; it is expected on every real-model run
// and is not a bug, so it is filtered out rather than muting console-error checks altogether.
const BENIGN_CONSOLE_ERROR = /VerifyEachNodeIsAssignedToAnEp/;

const test = base.extend<{ pageErrors: string[] }>({
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

test.describe("stub agent", () => {
  test("stub agent plays Snake: board advances, pause, speed and reset work", async ({
    page,
    pageErrors,
  }) => {
    await page.goto("/snake.html?model=stub&width=8&height=6&length=4&fps=60");
    await page.waitForSelector("body[data-ready]");

    await expect(page.locator("#state")).toHaveText("LIVE");
    await expect(page.locator("#executing")).not.toHaveText("—", { timeout: 5000 });

    await expect(page.locator("#length")).not.toHaveText("0", { timeout: 5000 });
    await expect
      .poll(async () => Number.parseFloat((await text(page, "rate")) ?? "0"), { timeout: 5000 })
      .toBeGreaterThan(0);

    await page.keyboard.press("Space");
    await expect(page.locator("#state")).toHaveText("PAUSED");
    await page.keyboard.press("Space");
    await expect(page.locator("#state")).toHaveText("LIVE");

    await page.keyboard.press("ArrowUp");
    await page.keyboard.press("ArrowUp");
    await expect(page.locator("#state")).toHaveText("LIVE");

    await page.locator("#max-speed").click();
    await expect(page.locator("#max-speed")).toHaveAttribute("aria-pressed", "true");

    await page.keyboard.press("r");
    await expect(page.locator("#round")).toHaveText("ROUND 02");

    await expect(page.locator("#engine")).toHaveText("STUB · planner");

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("the final board stays visible before the next round", async ({ page, pageErrors }) => {
    await page.goto("/snake.html?model=stub&width=4&height=4&length=2&fps=60&max-speed");
    await page.waitForSelector("body[data-ready]");

    await expect(page.locator("#state")).toHaveText("BOARD CLEAR", { timeout: 30_000 });
    await expect(page.locator("#length")).toHaveText("16");
    const clearedAt = await page.evaluate(() => performance.now());

    await expect.poll(async () => text(page, "round"), { timeout: 5000 }).toBe("ROUND 02");
    const resumedAt = await page.evaluate(() => performance.now());

    expect(resumedAt - clearedAt).toBeGreaterThanOrEqual(800);
    await expect(page.locator("#state")).toHaveText("LIVE");

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("unassisted mode runs without the shield", async ({ page, pageErrors }) => {
    await page.goto("/snake.html?model=stub&unassisted&width=8&height=6&length=4&fps=60");
    await page.waitForSelector("body[data-ready]");

    await expect(page.locator("#state")).toHaveText("LIVE");
    await expect(page.locator("#executing")).not.toHaveText("—", { timeout: 5000 });

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("a bad model URL surfaces an error", async ({ page }) => {
    await page.goto("/snake.html?model=/nonexistent/");

    await expect(page.locator("#loading")).toContainText("ERROR", { timeout: 15_000 });
    await expect(page.locator("#state")).toHaveText("ERROR");
  });
});

test.describe("real model", () => {
  const modelUrl = process.env.LAYA_MODEL_URL;

  test("real model plays 40 decisions without dying", async ({ page, pageErrors }) => {
    test.skip(!modelUrl, "set LAYA_MODEL_URL=<bundle directory URL> to run the real-model check");
    test.setTimeout(10 * 60 * 1000);

    await page.goto(`/snake.html?model=${encodeURIComponent(modelUrl!)}&max-speed`);

    await expect(page.locator("#engine")).toContainText("WebGPU", { timeout: 9 * 60 * 1000 });

    await expect
      .poll(async () => Number.parseInt((await text(page, "length")) ?? "0", 10), {
        timeout: 60_000,
      })
      .toBeGreaterThanOrEqual(8);

    await page.waitForTimeout(8000);

    await expect(page.locator("#round")).toHaveText("ROUND 01");

    expect(Number.parseFloat((await text(page, "inference")) ?? "")).toBeGreaterThan(0);
    expect(Number.parseFloat((await text(page, "rate")) ?? "")).toBeGreaterThan(0);

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });
});

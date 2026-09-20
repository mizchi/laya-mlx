import { expect, type Page, test } from "@playwright/test";

const text = (page: Page, id: string) => page.locator(`#${id}`).textContent();

test.describe("stub agent", () => {
  test("stub agent plays Snake: board advances, pause, speed and reset work", async ({ page }) => {
    const consoleErrors: string[] = [];
    const pageErrors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => pageErrors.push(String(error)));

    await page.goto("/snake.html?model=stub&width=8&height=6&length=4&fps=60");
    await page.waitForSelector("body[data-ready]");

    await expect(page.locator("#state")).toHaveText("LIVE");
    await expect(page.locator("#executing")).not.toHaveText("—", { timeout: 5000 });

    await page.waitForTimeout(1500);
    await expect(page.locator("#length")).not.toHaveText("0");
    const rate = Number.parseFloat((await text(page, "rate")) ?? "");
    expect(rate).toBeGreaterThan(0);

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

    expect(consoleErrors, consoleErrors.join("\n")).toEqual([]);
    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("the final board stays visible before the next round", async ({ page }) => {
    await page.goto("/snake.html?model=stub&width=4&height=4&length=2&fps=60&max-speed");
    await page.waitForSelector("body[data-ready]");

    await expect(page.locator("#state")).toHaveText("BOARD CLEAR", { timeout: 30_000 });
    await expect(page.locator("#length")).toHaveText("16");
    const clearedAt = await page.evaluate(() => performance.now());

    await expect.poll(async () => text(page, "round"), { timeout: 5000 }).toBe("ROUND 02");
    const resumedAt = await page.evaluate(() => performance.now());

    expect(resumedAt - clearedAt).toBeGreaterThanOrEqual(800);
    await expect(page.locator("#state")).toHaveText("LIVE");
  });

  test("unassisted mode runs without the shield", async ({ page }) => {
    await page.goto("/snake.html?model=stub&unassisted&width=8&height=6&length=4&fps=60");
    await page.waitForSelector("body[data-ready]");

    await expect(page.locator("#state")).toHaveText("LIVE");
    await expect(page.locator("#executing")).not.toHaveText("—", { timeout: 5000 });
  });

  test("a bad model URL surfaces an error", async ({ page }) => {
    await page.goto("/snake.html?model=/nonexistent/");

    await expect(page.locator("#loading")).toContainText("ERROR", { timeout: 15_000 });
    await expect(page.locator("#state")).toHaveText("ERROR");
  });
});

test.describe("real model", () => {
  const modelUrl = process.env.LAYA_MODEL_URL;

  test("real model plays 40 decisions without dying", async ({ page }) => {
    test.skip(!modelUrl, "set LAYA_MODEL_URL=<bundle directory URL> to run the real-model check");
    test.setTimeout(10 * 60 * 1000);

    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(String(error)));

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

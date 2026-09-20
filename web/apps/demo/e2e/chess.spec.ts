import { expect, test, text } from "./fixtures.ts";

test.describe("stub agent", () => {
  test("user moves and the stub AI replies", async ({ page, pageErrors }) => {
    await page.goto("/chess.html?model=stub");
    await page.waitForSelector("body[data-ready]");

    await expect(page.locator("#state")).toHaveText("LIVE");

    await page.locator("[data-square=e2]").click();
    await expect(page.locator("[data-square=e4]")).toHaveClass(/target/);
    await expect(page.locator("[data-square=e5]")).not.toHaveClass(/target/);

    await page.locator("[data-square=e4]").click();
    await expect(page.locator("#moves")).toContainText(/1\. e4 \S+/, { timeout: 5000 });

    await expect(page.locator("#executing")).not.toHaveText("—");
    const candidateCount = await page.locator("#candidates li").count();
    expect(candidateCount).toBeGreaterThanOrEqual(1);
    expect(candidateCount).toBeLessThanOrEqual(6);
    await expect(page.locator("#candidates li.best")).toHaveCount(1);

    await expect(page.locator("#engine")).toHaveText("STUB · planner");

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("illegal targets are ignored", async ({ page, pageErrors }) => {
    await page.goto("/chess.html?model=stub");
    await page.waitForSelector("body[data-ready]");

    await page.locator("[data-square=e2]").click();
    await expect(page.locator("[data-square=e2]")).toHaveClass(/selected/);

    // e5 is not a legal target for the pawn on e2 (empty square out of reach for a first move);
    // clicking it clears the selection rather than moving, so the move list stays empty.
    await page.locator("[data-square=e5]").click();

    await expect(page.locator("#moves")).toBeEmpty();
    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("switching sides makes the AI open and flips the board", async ({ page, pageErrors }) => {
    await page.goto("/chess.html?model=stub");
    await page.waitForSelector("body[data-ready]");

    await page.locator("#switch-sides").click();
    await expect(page.locator("#switch-sides")).toHaveText("Play as white");

    await expect(page.locator("#moves")).toContainText(/1\. \S+/, { timeout: 5000 });

    const a1Box = await page.locator("[data-square=a1]").boundingBox();
    const a8Box = await page.locator("[data-square=a8]").boundingBox();
    expect(a1Box).not.toBeNull();
    expect(a8Box).not.toBeNull();
    expect(a1Box!.y).toBeLessThan(a8Box!.y);

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("new game clears the board", async ({ page, pageErrors }) => {
    await page.goto("/chess.html?model=stub");
    await page.waitForSelector("body[data-ready]");

    await page.locator("[data-square=e2]").click();
    await page.locator("[data-square=e4]").click();
    await expect(page.locator("#moves")).toContainText(/1\. e4 \S+/, { timeout: 5000 });

    await page.locator("#new-game").click();

    await expect(page.locator("#moves")).toBeEmpty();
    await expect(page.locator("#state")).toHaveText("LIVE");
    await expect(page.locator("#executing")).toHaveText("—");

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("checkmate ends the game", async ({ page, pageErrors }) => {
    await page.goto(
      "/chess.html?model=stub&fen=" + encodeURIComponent("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    );
    await page.waitForSelector("body[data-ready]");

    await page.locator("[data-square=a1]").click();
    await page.locator("[data-square=a8]").click();

    await expect(page.locator("#status")).toHaveText("Checkmate — White wins");
    await expect(page.locator("#state")).toHaveText("GAME OVER");
    await expect(page.locator("[data-square=g8]")).toBeDisabled();

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });

  test("a bad model URL surfaces an error", async ({ page }) => {
    await page.goto("/chess.html?model=/nonexistent/");

    await expect(page.locator("#loading")).toContainText("ERROR", { timeout: 15_000 });
    await expect(page.locator("#state")).toHaveText("ERROR");
  });
});

test.describe("real model", () => {
  const modelUrl = process.env.LAYA_MODEL_URL;

  test("real model replies to 1. e4", async ({ page, pageErrors }) => {
    test.skip(!modelUrl, "set LAYA_MODEL_URL=<bundle directory URL> to run the real-model check");
    test.setTimeout(10 * 60 * 1000);

    await page.goto(`/chess.html?model=${encodeURIComponent(modelUrl!)}`);

    await expect(page.locator("#engine")).toContainText("WebGPU", { timeout: 9 * 60 * 1000 });

    await page.locator("[data-square=e2]").click();
    await page.locator("[data-square=e4]").click();

    await expect(page.locator("#moves")).toContainText(/1\. e4 \S+/, { timeout: 60_000 });

    expect(Number.parseFloat((await text(page, "inference")) ?? "")).toBeGreaterThan(0);

    expect(pageErrors, pageErrors.join("\n")).toEqual([]);
  });
});

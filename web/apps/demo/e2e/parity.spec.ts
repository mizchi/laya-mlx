import { expect, test } from "@playwright/test";

test("index page renders the heading and the parity link", async ({ page }) => {
  await page.goto("/index.html");
  await expect(page.locator("h1")).toHaveText("Laya in the browser");
  await expect(page.locator('a[href="parity.html"]')).toBeVisible();
});

const modelUrl = process.env.LAYA_MODEL_URL;

test("browser predictions match the Python fixtures", async ({ page }) => {
  test.skip(
    !modelUrl,
    "set LAYA_MODEL_URL=<bundle directory URL> to run the real-model parity check",
  );
  const lines: string[] = [];
  page.on("console", (message) => lines.push(message.text()));
  await page.goto(`/parity.html?model=${encodeURIComponent(modelUrl!)}`);
  await expect(page.locator("#status")).toContainText(/done:|ERROR/, { timeout: 14 * 60 * 1000 });
  const result = lines.find((line) => line.startsWith("[laya] RESULT "));
  expect(result, lines.join("\n")).toBeDefined();
  const summary = JSON.parse(result!.slice("[laya] RESULT ".length));
  expect(summary.provider).toBe("webgpu");
  expect(summary.argmax_agreements).toBe(summary.questions);
  expect(summary.questions).toBe(63);
  expect(summary.probability_max_abs_error).toBeLessThan(2e-2);
});

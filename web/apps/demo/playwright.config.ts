import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 15 * 60 * 1000,
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: {
    command: "pnpm vite --port 4173 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"],
        },
      },
    },
  ],
});

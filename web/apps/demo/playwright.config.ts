import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: {
    // Build then preview `dist/`, not `vite` (dev): dev serves ORT assets from node_modules and
    // the fixture straight out of `public/` (see vite.config.ts's dev `configureServer` hook),
    // neither of which exercises the `dist/ort/` copy, the `BASE_URL`-prefixed paths, or the
    // generateBundle cleanup that this app's build actually ships — only `dist/` does.
    command: "pnpm vite build && pnpm vite preview --port 4173 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--ignore-gpu-blocklist",
            // ANGLE's Metal backend is macOS-only; Linux CI would need a different backend
            // (e.g. "--use-angle=vulkan" or "--use-angle=swiftshader") and its own verification.
            ...(process.platform === "darwin" ? ["--use-angle=metal"] : []),
          ],
        },
      },
    },
  ],
});

import { defineConfig } from "vite";

export default defineConfig({
  server: {
    // The graph and its label list live beside the checkpoint that produced
    // them, not inside this app, so serving them means reading one directory up.
    fs: { allow: [".."] },
  },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});

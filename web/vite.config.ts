import { defineConfig } from "vite";

export default defineConfig({
  // Asset URLs are written relative to the page rather than to the origin root,
  // so a build works wherever it is served from: a site published under a path
  // as readily as one at a domain root. The default writes /assets/... into the
  // page, which asks the origin root for files that live one level down. That
  // failure cannot appear before deployment, because the dev server and the
  // preview server both serve at the root.
  base: "./",
  build: {
    // Emit every asset as its own file rather than inlining small ones as data
    // URLs. The label list is small enough to fall under the default threshold,
    // and inlining folds the class ordering into the bundle. It is written
    // beside the graph precisely so it stays a file that can be read on its own.
    assetsInlineLimit: 0,
  },
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

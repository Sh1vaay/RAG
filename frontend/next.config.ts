import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /**
   * The dashboard is a pure client-side SPA talking to FastAPI, so it exports to
   * static HTML/CSS/JS. `next build` writes `out/`, which FastAPI mounts — that
   * keeps the whole app a single service with no Node runtime in production.
   */
  output: "export",

  // Emit `out/<route>/index.html` so a plain static file server resolves paths
  // without rewrite rules.
  trailingSlash: true,

  // next/image optimisation needs a server; static export has none.
  images: { unoptimized: true },
};

export default nextConfig;

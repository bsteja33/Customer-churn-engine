import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_BUILD_TIME: new Date().toISOString().replace("T", " ").split(".")[0] + " UTC",
  },
  async rewrites() {
    // `BACKEND_PORT` is the FastAPI port. Defaults to 8000; can be
    // overridden independently of Next.js' own `PORT` (which `next dev`
    // reads for its own listen address).
    // `BACKEND_INTERNAL_URL` is the full override for split-host
    // deployments (e.g. `https://churn-api.onrender.com`). It wins
    // over `BACKEND_PORT` when both are set.
    const backendPort = process.env.BACKEND_PORT || "8000";
    const backend =
      process.env.BACKEND_INTERNAL_URL ||
      `http://127.0.0.1:${backendPort}`;
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/:path*`,
      },
      // The FastAPI Swagger UI at /docs is opened in a new tab from the
      // FE origin. Its relative fetch of /openapi.json would 404
      // against the FE; forward both to the backend so the docs page
      // resolves.
      {
        source: "/openapi.json",
        destination: `${backend}/openapi.json`,
      },
      {
        source: "/docs",
        destination: `${backend}/docs`,
      },
      {
        source: "/redoc",
        destination: `${backend}/redoc`,
      },
    ];
  },
};

export default nextConfig;

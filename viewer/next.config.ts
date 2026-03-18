import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/predictions/:path*",
        destination: "http://localhost:8011/predictions/:path*",
      },
      {
        source: "/api/dispatch/:path*",
        destination: "http://localhost:8011/dispatch/:path*",
      },
    ];
  },
};

export default nextConfig;

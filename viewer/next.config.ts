import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/predictions/:path*",
        destination: "http://localhost:8011/predictions/:path*",
      },
    ];
  },
};

export default nextConfig;

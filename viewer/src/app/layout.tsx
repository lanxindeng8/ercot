import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ERCOT Settlement Point Prices",
  description: "Real-time and Day-Ahead Market Settlement Point Prices from ERCOT",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

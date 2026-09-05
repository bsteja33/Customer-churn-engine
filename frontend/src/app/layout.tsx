import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AppShell } from "../components/AppShell";
import { ErrorBoundary } from "../components/ErrorBoundary";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Enterprise Churn Engine",
  description: "Next-generation predictive analytics and retention script generation.",
};

const BUILD_TIME = process.env.NEXT_PUBLIC_BUILD_TIME || "development";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="h-full bg-black text-white selection:bg-white selection:text-black overflow-x-hidden">
        <ErrorBoundary>
          <AppShell>
            {children}
            <footer className="w-full text-center py-4">
              <span className="text-[10px] text-white/20 tracking-wider font-mono">
                Build: {BUILD_TIME}
              </span>
            </footer>
          </AppShell>
        </ErrorBoundary>
      </body>
    </html>
  );
}

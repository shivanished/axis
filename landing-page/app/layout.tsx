import type { Metadata } from "next";
import { Outfit } from "next/font/google"; // using Outfit as a high-quality geometric fallback for Garet
import "./globals.css";

const garet = Outfit({
  subsets: ["latin"],
  variable: "--font-garet",
  weight: ["300", "400", "500", "700", "900"],
});

export const metadata: Metadata = {
  title: "Axis | Design at the Speed of Thought",
  description: "Transform your webcam into a 3D input device.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${garet.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

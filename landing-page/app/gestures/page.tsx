"use client";

import { motion } from "framer-motion";
import { ChevronLeft } from "lucide-react";
import Link from "next/link";
import { useState, useEffect } from "react";

export default function GesturesPage() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 20,
        y: (e.clientY / window.innerHeight - 0.5) * 20,
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  useEffect(() => {
    document.body.style.overflow = "auto";
    return () => {
      document.body.style.overflow = "hidden";
    };
  }, []);

  return (
    <main className="relative flex min-h-screen w-full flex-col items-center justify-start py-8 bg-black text-white selection:bg-white/30 selection:text-white">
      {/* Cinematic Background Layer */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <motion.div
          className="absolute inset-[-5%] w-[110%] h-[110%] bg-cover bg-center opacity-60"
          style={{
            backgroundImage: 'url("/image.png")',
          }}
          animate={{
            x: mousePosition.x,
            y: mousePosition.y,
            scale: [1, 1.05, 1],
          }}
          transition={{
            scale: {
              duration: 20,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "easeInOut",
            },
            x: { type: "spring", stiffness: 50, damping: 30 },
            y: { type: "spring", stiffness: 50, damping: 30 },
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-transparent to-black/90" />
        <div className="absolute inset-0 bg-gradient-to-r from-black/60 via-transparent to-black/60" />
        <div className="absolute inset-0 opacity-30 mix-blend-screen pointer-events-none">
          <motion.div
            className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-500/20 rounded-full blur-[100px]"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 8, repeat: Infinity }}
          />
          <motion.div
            className="absolute bottom-1/3 right-1/3 w-96 h-96 bg-indigo-500/10 rounded-full blur-[120px]"
            animate={{ scale: [1.2, 1, 1.2], opacity: [0.2, 0.4, 0.2] }}
            transition={{ duration: 10, repeat: Infinity, delay: 2 }}
          />
        </div>
      </div>

      <div className="z-10 flex w-full max-w-6xl flex-col items-center justify-start px-4">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8 w-full"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-[10px] font-bold text-neutral-400 hover:text-white transition-colors uppercase tracking-[0.25em]"
          >
            <ChevronLeft className="w-3 h-3" /> Return
          </Link>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="mb-8 font-sans text-5xl font-semibold tracking-tighter text-white md:text-7xl leading-none select-none"
          style={{ textShadow: "0 0 30px rgba(255,255,255,0.2)" }}
        >
          Gestures
        </motion.h1>

        {/* Gesture Images */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="w-full max-w-5xl mb-32"
        >
          <div className="flex w-full flex-col gap-8">
            {/* Motion Gestures */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="group relative overflow-hidden rounded-[2rem] border border-white/10 bg-black/40 backdrop-blur-2xl transition-all duration-500 hover:border-white/20 hover:bg-black/60"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
              <div className="relative z-10 p-6">
                <h3 className="mb-4 text-lg font-light text-white tracking-wide">
                  Motion Gestures
                </h3>
                <div className="relative overflow-hidden rounded-xl">
                  <img
                    src="/gestures1.jpg"
                    alt="Motion Gestures"
                    className="w-full h-auto object-cover transition-transform duration-500 group-hover:scale-105"
                  />
                </div>
              </div>
            </motion.div>

            {/* Static Gestures */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.6 }}
              className="group relative overflow-hidden rounded-[2rem] border border-white/10 bg-black/40 backdrop-blur-2xl transition-all duration-500 hover:border-white/20 hover:bg-black/60"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
              <div className="relative z-10 p-6">
                <h3 className="mb-4 text-lg font-light text-white tracking-wide">
                  Static Gestures
                </h3>
                <div className="relative overflow-hidden rounded-xl">
                  <img
                    src="/gestures2.jpg"
                    alt="Static Gestures"
                    className="w-full h-auto object-cover transition-transform duration-500 group-hover:scale-105"
                  />
                </div>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </div>

      {/* Footer Status / Copyright */}
      <motion.div
        className="absolute bottom-8 flex w-full max-w-6xl justify-between px-10 text-[9px] uppercase tracking-[0.2em] text-neutral-500 font-mono mix-blend-plus-lighter"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
      >
        <div className="flex items-center gap-2">
          <span>© 2025 Axis Systems</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="h-1 w-1 rounded-full bg-white animate-pulse"></span>
          <span>Status: Online</span>
        </div>
      </motion.div>
    </main>
  );
}


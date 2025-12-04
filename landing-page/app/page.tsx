"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence, useAnimation } from "framer-motion";
import { Scan, Box, MousePointer2, Download, ChevronLeft } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function Home() {
  const [showSolution, setShowSolution] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 20, // -10 to 10
        y: (e.clientY / window.innerHeight - 0.5) * 20, // -10 to 10
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  // Staggered text variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20, filter: "blur(10px)" },
    visible: {
      opacity: 1,
      y: 0,
      filter: "blur(0px)",
      transition: { duration: 1.2, ease: [0.16, 1, 0.3, 1] as const },
    },
  };

  const features = [
    {
      title: "Detection",
      description:
        "CNN-powered inference engine identifies 21 skeletal keypoints across your hand.",
      icon: Scan,
    },
    {
      title: "Projection",
      description:
        "Projection matrices convert 2D-coordinates into calibrated 3D-spatial trajectories.",
      icon: Box,
    },
    {
      title: "Actuation",
      description:
        "System-level integration translates hand kinematics into precise cursor movements.",
      icon: MousePointer2,
    },
  ];

  return (
    <main className="relative flex min-h-screen w-full flex-col items-center justify-center overflow-hidden bg-black text-white selection:bg-white/30 selection:text-white">
      {/* Cinematic Background Layer */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        {/* Image Container with Parallax & Breathing */}
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

        {/* Atmospheric Overlay Gradients */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-transparent to-black/90" />
        <div className="absolute inset-0 bg-gradient-to-r from-black/60 via-transparent to-black/60" />

        {/* Subtle Floating Particles / Dust (CSS) */}
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

      <div className="z-10 flex w-full max-w-6xl flex-col items-center justify-center px-4">
        <AnimatePresence mode="wait">
          {!showSolution ? (
            <motion.div
              key="hero"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              exit={{
                opacity: 0,
                y: -20,
                filter: "blur(10px)",
                transition: { duration: 0.5 },
              }}
              className="flex flex-col items-center text-center mix-blend-difference" // Experimental blend mode for text
            >
              <motion.h1
                variants={itemVariants}
                className="font-sans text-7xl font-semibold tracking-tighter text-white md:text-9xl leading-none select-none"
                style={{ textShadow: "0 0 30px rgba(255,255,255,0.2)" }}
              >
                axis
              </motion.h1>
              <motion.div variants={itemVariants} className="relative">
                <p className="mt-6 max-w-lg text-base font-light text-white/80 md:text-xl tracking-[0.2em] uppercase">
                  design at the speed of thought
                </p>
                {/* Animated Underline */}
                <motion.div
                  className="absolute -bottom-4 left-1/2 h-[1px] bg-gradient-to-r from-transparent via-white/50 to-transparent"
                  initial={{ width: 0, x: "-50%" }}
                  animate={{ width: "100%", x: "-50%" }}
                  transition={{ delay: 1, duration: 1.5, ease: "easeInOut" }}
                />
              </motion.div>

              <motion.div variants={itemVariants} className="mt-20">
                <button
                  onClick={() => setShowSolution(true)}
                  className="group relative cursor-pointer overflow-hidden rounded-full border border-white/20 bg-white/5 px-10 py-4 text-sm backdrop-blur-md transition-all hover:border-white/40 hover:bg-white/10 hover:shadow-[0_0_40px_-10px_rgba(255,255,255,0.2)] active:scale-95"
                >
                  <a
                    href="https://met-glasses.s3.us-east-2.amazonaws.com/MetGlasses.app.zip"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="relative z-10 flex items-center gap-3 text-neutral-200 transition-colors group-hover:text-white"
                  >
                    <span className="uppercase tracking-[0.15em] text-xs font-semibold">
                      Download Now
                    </span>
                    <Download className="h-4 w-4 transition-transform duration-500 group-hover:translate-y-1" />
                  </a>
                  {/* Button Shine Effect */}
                  <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/10 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
                </button>
              </motion.div>
            </motion.div>
          ) : (
            <motion.div
              key="solution"
              initial={{ opacity: 0, scale: 0.9, filter: "blur(20px)" }}
              animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
              exit={{ opacity: 0, scale: 1.05, filter: "blur(10px)" }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] as const }}
              className="flex w-full flex-col items-center"
            >
              <motion.button
                onClick={() => setShowSolution(false)}
                className="mb-16 flex items-center gap-2 text-[10px] font-bold text-neutral-400 hover:text-white transition-colors uppercase tracking-[0.25em]"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <ChevronLeft className="w-3 h-3" /> Return
              </motion.button>

              <div className="grid w-full grid-cols-1 gap-8 md:grid-cols-3 perspective-[1000px]">
                {features.map((feature, index) => (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 40, rotateX: -10 }}
                    animate={{ opacity: 1, y: 0, rotateX: 0 }}
                    transition={{
                      delay: index * 0.2 + 0.3,
                      duration: 0.8,
                      ease: "easeOut",
                    }}
                    className="group relative overflow-hidden rounded-[2rem] border border-white/10 bg-black/40 p-10 backdrop-blur-2xl transition-all duration-500 hover:border-white/20 hover:bg-black/60 hover:-translate-y-2 hover:shadow-[0_20px_40px_-10px_rgba(0,0,0,0.5)]"
                  >
                    {/* Card Inner Gradient */}
                    <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />

                    <div className="relative z-10 mb-8 inline-flex rounded-2xl border border-white/10 bg-white/5 p-4 text-white shadow-lg group-hover:border-white/20 group-hover:bg-white/10 transition-colors">
                      <feature.icon className="h-6 w-6 stroke-[1.5]" />
                    </div>
                    <h3 className="relative z-10 mb-4 text-2xl font-light text-white tracking-wide">
                      {feature.title}
                    </h3>
                    <p className="relative z-10 text-sm leading-relaxed text-neutral-400 group-hover:text-neutral-300 font-light tracking-wide">
                      {feature.description}
                    </p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer Status / Copyright */}
      <motion.div
        className="absolute bottom-8 flex w-full max-w-6xl justify-between px-10 text-[9px] uppercase tracking-[0.2em] text-neutral-500 font-mono mix-blend-plus-lighter"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
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

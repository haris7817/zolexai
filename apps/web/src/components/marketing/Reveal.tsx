"use client";

import { useEffect, useRef, useState, type ElementType, type ReactNode } from "react";

/**
 * Scroll-triggered entrance for a landing-page block.
 *
 * IntersectionObserver plus two CSS rules (see `tokens.css`) — no animation
 * dependency, nothing running on the scroll event, and each element is
 * unobserved the moment it has appeared, so a long page costs a handful of
 * one-shot callbacks rather than continuous work.
 *
 * Deliberately one-way: content that fades out again when scrolled past is a
 * distraction on a marketing page, and re-animating on every pass is the
 * cheapest way to make a premium page feel gimmicky.
 *
 * Rendering starts hidden, which puts a real obligation on this component:
 * anything that stops the observer from running must still end with visible
 * content. Reduced motion is handled in CSS, absent-observer support is
 * handled below, and the page carries a <noscript> fallback for JS being off
 * entirely.
 */
export function Reveal({
  children,
  as: Tag = "div",
  delay = 0,
  className,
  id,
}: {
  children: ReactNode;
  /** The element to render — a section entrance should still be a <section>. */
  as?: ElementType;
  /** Stagger, in milliseconds, for siblings revealed as a group. */
  delay?: number;
  className?: string;
  id?: string;
}) {
  const ref = useRef<HTMLElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    if (typeof IntersectionObserver === "undefined") {
      setShown(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setShown(true);
        observer.disconnect();
      },
      {
        // Fires a little before the block reaches the viewport, so the
        // entrance is under way by the time it is properly in view rather
        // than starting visibly late.
        rootMargin: "0px 0px -12% 0px",
        threshold: 0.05,
      },
    );

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <Tag
      ref={ref}
      id={id}
      className={className}
      data-zx-reveal={shown ? "shown" : "pending"}
      style={delay ? ({ "--zx-reveal-delay": `${delay}ms` } as React.CSSProperties) : undefined}
    >
      {children}
    </Tag>
  );
}

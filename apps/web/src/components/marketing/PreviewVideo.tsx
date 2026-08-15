"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

type PreviewVideoProps = {
  src: string;
  label: string;
  interaction: "hover" | "manual";
  className?: string;
};

/**
 * Lazy, silent landing-page video preview.
 *
 * The source is withheld until the preview approaches the viewport. Desktop
 * cards play only while hovered; touch devices retain a stable first frame.
 * The hero is always manual. Both modes pause when scrolled out of view and
 * leave the existing parent background visible if media loading fails.
 */
function PreviewVideo({ src, label, interaction, className }: PreviewVideoProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [shouldLoad, setShouldLoad] = useState(false);
  const [inViewport, setInViewport] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);

  const pause = useCallback((reset = false) => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    if (reset && video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      video.currentTime = 0;
    }
  }, []);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    if (typeof IntersectionObserver === "undefined") {
      setShouldLoad(true);
      setInViewport(true);
      return;
    }

    const loadObserver = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setShouldLoad(true);
        loadObserver.disconnect();
      },
      { rootMargin: "240px 0px", threshold: 0.01 },
    );
    const visibilityObserver = new IntersectionObserver(
      ([entry]) => setInViewport(entry.isIntersecting),
      { threshold: 0.01 },
    );

    loadObserver.observe(element);
    visibilityObserver.observe(element);
    return () => {
      loadObserver.disconnect();
      visibilityObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReducedMotion(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (!inViewport || (interaction === "hover" && reducedMotion)) pause();
  }, [inViewport, interaction, pause, reducedMotion]);

  const play = useCallback(() => {
    if (failed || !inViewport) return;
    void videoRef.current?.play().catch(() => setPlaying(false));
  }, [failed, inViewport]);

  const hoverAllowed = useCallback(
    () =>
      !reducedMotion &&
      window.matchMedia("(hover: hover) and (pointer: fine)").matches,
    [reducedMotion],
  );

  return (
    <div
      ref={containerRef}
      className={cn("overflow-hidden", className)}
      onPointerEnter={() => {
        if (interaction === "hover" && hoverAllowed()) play();
      }}
      onPointerLeave={() => {
        if (interaction === "hover" && hoverAllowed()) pause(true);
      }}
    >
      {shouldLoad && !failed ? (
        <video
          ref={videoRef}
          src={src}
          muted
          loop
          playsInline
          preload="metadata"
          aria-label={`${label} silent preview`}
          className="h-full w-full object-cover"
          onLoadedData={() => setReady(true)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={() => {
            pause(true);
            setFailed(true);
          }}
        />
      ) : null}

      {interaction === "manual" ? (
        <button
          type="button"
          onClick={() => {
            if (playing) pause();
            else play();
          }}
          disabled={!ready || failed}
          aria-label={`${playing ? "Pause" : "Play"} ${label}`}
          title={`${playing ? "Pause" : "Play"} ${label}`}
          className={cn(
            "absolute top-1/2 left-1/2 flex h-[58px] w-[58px] -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full",
            "cursor-pointer border border-white/25 bg-black/25 text-white backdrop-blur-[6px] transition-colors duration-150",
            "hover:border-zx-border-active hover:bg-black/45 disabled:cursor-default disabled:opacity-70",
          )}
        >
          <span className={playing ? "" : "ml-[3px]"}>
            <Icon name={playing ? "pause" : "play"} size={24} />
          </span>
        </button>
      ) : null}
    </div>
  );
}

export function ToolPreviewVideo({ src, label }: { src: string; label: string }) {
  return (
    <PreviewVideo
      src={src}
      label={label}
      interaction="hover"
      className="absolute inset-0"
    />
  );
}

export function HeroPreviewVideo({ src, label }: { src: string; label: string }) {
  return (
    <PreviewVideo
      src={src}
      label={label}
      interaction="manual"
      className="absolute inset-0"
    />
  );
}

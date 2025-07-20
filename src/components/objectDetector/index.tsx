import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { ObjectDetection } from "@tensorflow-models/coco-ssd";

interface Prediction {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

export function ObjectDetector() {
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [model, setModel] = useState<ObjectDetection | null>(null);

  // Kamera-Modus-Auswahl
  const CameraSelector = () => (
    <div className="absolute top-2 right-2 z-20 flex space-x-2">
      <button
        onClick={() => setFacingMode("user")}
        className={`px-2 py-1 rounded ${facingMode === "user" ? "bg-white text-black" : "bg-gray-700 text-white"}`}
      >
        Front
      </button>
      <button
        onClick={() => setFacingMode("environment")}
        className={`px-2 py-1 rounded ${facingMode === "environment" ? "bg-white text-black" : "bg-gray-700 text-white"}`}
      >
        Rück
      </button>
    </div>
  );

  // WebGL backend und Modell laden
  useEffect(() => {
    async function setup() {
      await tf.setBackend("webgl");
      await tf.ready();
      const loadedModel = await cocoSsd.load();
      setModel(loadedModel);
    }
    setup();
  }, []);

  // Kamera aktivieren (abhängig vom facingMode)
  useEffect(() => {
    async function enableCamera() {
      if (!videoRef.current) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    }
    enableCamera();
  }, [facingMode]);

  // Erkennung alle 100ms mit Skalierung basierend auf angezeigter Größe
  useEffect(() => {
    if (!model || !videoRef.current || !containerRef.current) return;
    const interval = setInterval(async () => {
      const video = videoRef.current!;
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      const { width: dispW, height: dispH } = video.getBoundingClientRect();
      const { videoWidth: vidW, videoHeight: vidH } = video;
      if (vidW === 0 || vidH === 0) return;
      const preds = await model.detect(video);
      const scaled = preds.map(p => {
        const [x, y, w, h] = p.bbox;
        return {
          class: p.class,
          score: p.score,
          bbox: [
            (x / vidW) * dispW,
            (y / vidH) * dispH,
            (w / vidW) * dispW,
            (h / vidH) * dispH,
          ] as [number, number, number, number],
        };
      });
      setPredictions(scaled);
    }, 100);
    return () => clearInterval(interval);
  }, [model]);

  return (
    <div ref={containerRef} className="relative w-full h-screen overflow-hidden bg-black">
      {/* Kamera-Auswahl */}
      <CameraSelector />
      <video
        ref={videoRef}
        className="absolute inset-0 m-auto w-full h-full object-contain"
        playsInline
        muted
      />
      {predictions.map((prediction, idx) => {
        const [x, y, w, h] = prediction.bbox;
        return (
          <div
            key={idx}
            className="absolute border-2 border-green-500"
            style={{ left: x, top: y, width: w, height: h }}
          >
            <span className="absolute -top-6 left-0 bg-green-500 text-white text-xs px-1">
              {prediction.class} {(prediction.score * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

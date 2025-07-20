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
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [model, setModel] = useState<ObjectDetection | null>(null);

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

  // Kamera aktivieren
  useEffect(() => {
    async function enableCamera() {
      if (!videoRef.current) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    }
    enableCamera();
  }, []);

  // Erkennung alle 100ms mit Skalierung basierend auf angezeigter Größe
  useEffect(() => {
    if (!model || !videoRef.current || !containerRef.current) return;
    const interval = setInterval(async () => {
      const video = videoRef.current!;
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      // Displayed dimensions (CSS)
      const { width: dispW, height: dispH } = video.getBoundingClientRect();
      // Roh-Videodaten-Dimensionen
      const { videoWidth: vidW, videoHeight: vidH } = video;
      if (vidW === 0 || vidH === 0) return;
      // Erkennung auf dem Video-Element
      const preds = await model.detect(video);
      // Normalisiere Bounding Boxes auf angezeigte Größe
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
    // Container füllt Bildschirm, Video zentriert und passt sich entweder Breite oder Höhe an
    <div ref={containerRef} className="relative w-full h-screen overflow-hidden bg-black">
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

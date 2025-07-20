import React, { useRef, useState, ChangeEvent } from "react";

import "@tensorflow/tfjs-backend-cpu";
// import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
// import { Roboflow } from "roboflow/js";
import { ObjectDetection } from "@tensorflow-models/coco-ssd";

interface Prediction {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

export function ObjectDetector() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const [imgData, setImgData] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setLoading] = useState(false);

  const isEmptyPredictions = !predictions || predictions.length === 0;

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const normalizePredictions = (
    predictions: cocoSsd.DetectedObject[],
    imgSize: { width: number; height: number }
  ): Prediction[] => {
    if (!predictions || !imgSize || !imageRef.current) return [];

    const imgWidth = imageRef.current.width;
    const imgHeight = imageRef.current.height;

    return predictions.map((prediction) => {
      const [oldX, oldY, oldWidth, oldHeight] = prediction.bbox;

      const x = (oldX * imgWidth) / imgSize.width;
      const y = (oldY * imgHeight) / imgSize.height;
      const width = (oldWidth * imgWidth) / imgSize.width;
      const height = (oldHeight * imgHeight) / imgSize.height;

      return {
        class: prediction.class,
        score: prediction.score,
        bbox: [x, y, width, height],
      };
    });
  };

  const detectObjectsOnImage = async (
    imageElement: HTMLImageElement,
    imgSize: { width: number; height: number }
  ) => {
    const model: ObjectDetection = await cocoSsd.load();
    const predictions = await model.detect(imageElement, 6);
    const normalized = normalizePredictions(predictions, imgSize);
    setPredictions(normalized);
    console.log("Predictions: ", predictions);
  };

  const readImage = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.onload = () => resolve(fileReader.result as string);
      fileReader.onerror = () => reject(fileReader.error);
      fileReader.readAsDataURL(file);
    });
  };

  const onSelectImage = async (e: ChangeEvent<HTMLInputElement>) => {
    setPredictions([]);
    setLoading(true);

    const file = e.target.files?.[0];
    if (!file) return;

    const imgData = await readImage(file);
    setImgData(imgData);

    const imageElement = document.createElement("img");
    imageElement.src = imgData;

    imageElement.onload = async () => {
      const imgSize = {
        width: imageElement.width,
        height: imageElement.height,
      };
      await detectObjectsOnImage(imageElement, imgSize);
      setLoading(false);
    };
  };

  return (
    <div className="flex flex-col items-center">
      <div className="min-w-[200px] h-[700px] border-4 border-white rounded-md flex items-center justify-center relative">
        {imgData && (
          <img
            src={imgData}
            ref={imageRef}
            alt="Target"
            className="h-full object-contain"
          />
        )}
        {!isEmptyPredictions &&
          predictions.map((prediction, idx) => (
            <div
              key={idx}
              className="absolute border-4 border-green-500"
              style={{
                left: `${prediction.bbox[0]}px`,
                top: `${prediction.bbox[1]}px`,
                width: `${prediction.bbox[2]}px`,
                height: `${prediction.bbox[3]}px`,
              }}
            >
              <div className="absolute -top-6 left-0 text-green-500 font-medium text-base">
                {`${prediction.class} ${(prediction.score * 100).toFixed(1)}%`}
              </div>
            </div>
          ))}
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={onSelectImage}
        className="hidden"
      />

      <button
        onClick={openFilePicker}
        className="mt-8 px-4 py-2 border-2 border-transparent bg-white text-[#0a0f22] text-lg font-medium transition-all duration-200 hover:bg-transparent hover:border-white hover:text-white"
      >
        {isLoading ? "Recognizing..." : "Select Image"}
      </button>
    </div>
  );
}

import "./App.css";
import { ObjectDetector } from "./components/objectDetector";

function App() {
  return (
    <div className="w-full h-full bg-[#1c2127] flex flex-col items-center justify-center text-white">
      <ObjectDetector />
    </div>
  );
}

export default App;

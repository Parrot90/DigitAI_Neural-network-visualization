'use client';

import { useDigitAi } from "@/hooks/use-digit-ai";
import ConfigPanel from "./config-panel";
import InferenceCanvas from "./inference-canvas";
import ModelVisualization from "./model-visualization";
import PredictionChart from "./prediction-chart";
import TrainingMonitor from "./training-monitor";
import { Skeleton } from "../ui/skeleton";
import { Separator } from "../ui/separator";
import { Button } from "../ui/button";
import { Eraser } from "lucide-react";

export default function DigitAiClient() {
  const manager = useDigitAi();

  const handleClearCanvas = () => {
    if (manager.canvasRef.current) {
      manager.canvasRef.current.clearCanvas();
    }
  };

  if (!manager.isReady) {
    return (
       <div className="w-full h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Skeleton className="h-16 w-16 rounded-full" />
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-4 w-48" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen p-4 gap-4">
      {/* Top Section */}
      <div className="flex-none grid md:grid-cols-2 gap-4 h-[40vh]">
        <div className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-muted-foreground pl-2">Inference</h2>
            <div className="relative flex-grow rounded-lg border bg-card p-4">
              <InferenceCanvas
                  onDrawEnd={manager.handleDraw}
                  onClear={manager.handleClear}
                  ref={manager.canvasRef}
                  isTraining={manager.isTraining}
              />
              <Button variant="ghost" size="sm" onClick={handleClearCanvas} disabled={manager.isTraining} className="absolute top-2 right-2 bg-background/50 backdrop-blur-sm">
                <Eraser className="w-4 h-4 mr-2" />
                Clear
              </Button>
            </div>
        </div>
        <div className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-muted-foreground pl-2">Prediction</h2>
            <div className="flex-grow rounded-lg border bg-card p-4">
                <PredictionChart 
                  predictions={manager.predictions} 
                  inferenceMetrics={manager.inferenceMetrics}
                />
            </div>
        </div>
      </div>

      <Separator />

      {/* Bottom Section */}
      <div className="flex-grow flex flex-col gap-2 min-h-0">
        <h2 className="text-sm font-semibold text-muted-foreground pl-2">Model Playground</h2>
        <div className="flex-grow rounded-lg border bg-card flex flex-col p-4 gap-4">
          {/* Main content */}
          <div className="flex-grow flex flex-col md:flex-row gap-4 min-h-0">
            <div className="w-full md:w-2/3 h-full min-h-[200px] md:min-h-full">
              <ModelVisualization 
                layersConfig={manager.config.layers} 
                activations={manager.activations}
                layerStats={manager.layerStats}
                onAddLayer={() => manager.addLayer({neurons: 64})}
                onRemoveLayer={manager.removeLayer}
                onNeuronCountChange={manager.updateLayerNeurons}
              />
            </div>
            <div className="w-full md:w-1/3 h-full min-h-[200px] md:min-h-full">
              <TrainingMonitor trainingHistory={manager.trainingHistory} />
            </div>
          </div>
          {/* Footer config panel */}
          <div className="flex-none">
            <ConfigPanel useDigitAiManager={manager} />
          </div>
        </div>
      </div>
    </div>
  );
}

'use client';

import React from 'react';
import { cn } from "@/lib/utils";
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Zap, Target, TrendingUp } from 'lucide-react';

interface Prediction {
  digit: number;
  confidence: number;
}

interface InferenceMetrics {
  confidence: number;
  predictedDigit: number;
  timestamp: number;
}

interface PredictionChartProps {
  predictions: Prediction[];
  inferenceMetrics?: InferenceMetrics | null;
}

export default function PredictionChart({ predictions, inferenceMetrics }: PredictionChartProps) {
  const sortedPredictions = [...predictions].sort((a, b) => b.confidence - a.confidence);
  const topPrediction = sortedPredictions[0];
  const secondPrediction = sortedPredictions[1];
  const confidenceGap = topPrediction ? topPrediction.confidence - (secondPrediction?.confidence || 0) : 0;
  const hasPredictions = predictions.some(p => p.confidence > 0.001);
  
  // Determine confidence level with enhanced feedback
  const getConfidenceLevel = (confidence: number, gap: number) => {
    if (confidence > 0.9 && gap > 0.5) 
      return { level: 'Very High', color: 'bg-emerald-500', variant: 'default' as const, icon: '🎯' };
    if (confidence > 0.8 && gap > 0.4) 
      return { level: 'High', color: 'bg-green-500', variant: 'default' as const, icon: '✅' };
    if (confidence > 0.6 && gap > 0.2) 
      return { level: 'Medium', color: 'bg-yellow-500', variant: 'secondary' as const, icon: '⚠️' };
    if (confidence > 0.4) 
      return { level: 'Low', color: 'bg-orange-500', variant: 'secondary' as const, icon: '❓' };
    return { level: 'Very Low', color: 'bg-red-500', variant: 'destructive' as const, icon: '🤔' };
  };

  const confidenceInfo = topPrediction ? getConfidenceLevel(topPrediction.confidence, confidenceGap) : null;

  if (!hasPredictions) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-center">
        <div className="text-4xl mb-4">✏️</div>
        <p className="text-muted-foreground">Draw a digit to see predictions</p>
        <p className="text-xs text-muted-foreground/70 mt-1">Real-time neural network inference</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col gap-3">
      {/* Header with metrics */}
      {inferenceMetrics && (
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4" />
            <span className="text-sm font-medium">
              Predicted: {inferenceMetrics.predictedDigit}
            </span>
            {confidenceInfo && (
              <span className="text-lg">{confidenceInfo.icon}</span>
            )}
          </div>
          
          {confidenceInfo && (
            <Badge variant={confidenceInfo.variant} className="text-xs">
              <Zap className="w-3 h-3 mr-1" />
              {confidenceInfo.level}
            </Badge>
          )}
        </div>
      )}

      {/* Certainty indicators */}
      {confidenceInfo && (
        <div className="mb-3 p-2 rounded-lg bg-muted/30">
          <div className="flex items-center justify-between text-xs">
            <span>Confidence Gap:</span>
            <span className={cn(
              "font-medium",
              confidenceGap > 0.5 ? "text-green-600" : 
              confidenceGap > 0.3 ? "text-yellow-600" : "text-red-600"
            )}>
              {(confidenceGap * 100).toFixed(1)}%
            </span>
          </div>
          <Progress 
            value={confidenceGap * 100} 
            className="h-2 mt-1"
          />
          {confidenceGap < 0.2 && (
            <p className="text-xs text-muted-foreground mt-1">
              Low certainty - consider redrawing more clearly
            </p>
          )}
        </div>
      )}

      {/* Main prediction bars */}
      <div className="flex-1 flex items-end justify-around gap-1 min-h-0">
        {predictions.sort((a, b) => a.digit - b.digit).map((prediction) => {
          const isTop = prediction.digit === topPrediction?.digit;
          const isSecond = prediction.digit === secondPrediction?.digit;
          const height = prediction.confidence * 100;
          
          return (
            <div key={prediction.digit} className="flex flex-col items-center justify-end h-full flex-1 min-w-0">
              {/* Confidence percentage (only show for top 3) */}
              {(isTop || isSecond || sortedPredictions.indexOf(prediction) === 2) && (
                <div className={cn(
                  "text-xs mb-1 font-medium transition-colors",
                  isTop ? "text-primary" : "text-muted-foreground"
                )}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </div>
              )}
              
              {/* Bar container */}
              <div className="w-full h-full bg-accent/20 rounded-t-sm flex items-end relative overflow-hidden">
                {/* Main bar */}
                <div 
                  className={cn(
                    "w-full rounded-t-sm transition-all duration-500 ease-out relative",
                    isTop ? "bg-primary" : isSecond ? "bg-primary/70" : "bg-accent",
                    isTop && "shadow-lg shadow-primary/20"
                  )}
                  style={{ height: `${height}%` }}
                >
                  {/* Glow effect for top prediction */}
                  {isTop && height > 10 && (
                    <div className="absolute inset-0 bg-primary/30 rounded-t-sm animate-pulse" />
                  )}
                  
                  {/* Ranking indicator for top 3 */}
                  {(isTop || isSecond || sortedPredictions.indexOf(prediction) === 2) && height > 20 && (
                    <div className="absolute top-1 right-1">
                      <div className={cn(
                        "w-2 h-2 rounded-full",
                        isTop ? "bg-yellow-400" : isSecond ? "bg-gray-400" : "bg-orange-400"
                      )} />
                    </div>
                  )}
                </div>
              </div>
              
              {/* Digit label */}
              <div className={cn(
                "mt-2 text-sm border rounded-md w-full text-center py-1 transition-colors",
                isTop 
                  ? "border-primary text-primary bg-primary/10 font-semibold" 
                  : "border-border text-muted-foreground"
              )}>
                {prediction.digit}
              </div>
            </div>
          );
        })}
      </div>

      {/* Additional metrics */}
      {topPrediction && confidenceGap > 0 && (
        <div className="flex-none space-y-2 pt-2 border-t">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              Confidence Gap:
            </span>
            <span className="font-mono">{(confidenceGap * 100).toFixed(1)}%</span>
          </div>
          
          <Progress 
            value={confidenceGap * 100} 
            className="h-2" 
          />
          
          <div className="text-xs text-muted-foreground">
            {confidenceGap > 0.5 
              ? "Very certain prediction" 
              : confidenceGap > 0.3 
                ? "Moderately certain" 
                : "Low certainty - ambiguous input"
            }
          </div>
        </div>
      )}
    </div>
  );
}

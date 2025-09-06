
'use client';

import React, { useState, useMemo } from 'react';
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Plus, X, Zap, Activity, Eye, Info } from "lucide-react";
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';

interface LayerConfig {
  neurons: number;
}

interface LayerStats {
  layerIndex: number;
  totalNeurons: number;
  activeNeurons: number;
  activationRate: number;
  avgActivation: number;
  maxActivation: number;
  dominantFeatures: { idx: number; activation: number; }[];
}

interface ModelVisualizationProps {
  layersConfig: LayerConfig[];
  activations: number[][];
  layerStats?: LayerStats[];
  onAddLayer: () => void;
  onRemoveLayer: (index: number) => void;
  onNeuronCountChange: (index: number, neurons: number) => void;
}

const MAX_NODES_TO_RENDER = 8;
const NODE_SIZE = "h-10 w-10";

const Neuron = ({ 
  activation, 
  isDominant, 
  neuronIndex, 
  layerIndex,
  onClick 
}: { 
  activation: number;
  isDominant: boolean;
  neuronIndex: number;
  layerIndex: number;
  onClick?: () => void;
}) => {
  const intensity = Math.min(Math.abs(activation), 1);
  const isActive = intensity > 0.05; // Lower threshold for better visibility
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              NODE_SIZE, 
              "rounded-full border-3 transition-all duration-500 flex-shrink-0 cursor-pointer relative",
              "hover:scale-125 hover:z-10 transform",
              // Base styling with better contrast using theme colors
              "bg-muted border-border",
              // Active neuron styling - much brighter using theme primary
              isActive && 'border-primary bg-primary/80 shadow-lg',
              // Dominant neuron styling - very bright using accent colors
              isDominant && 'ring-4 ring-accent ring-opacity-90 border-accent bg-accent/80'
            )}
            style={{
              boxShadow: isActive 
                ? `0 0 ${12 + intensity * 24}px hsl(var(--primary) / ${0.6 + intensity * 0.4})` 
                : '0 0 4px hsl(var(--muted-foreground) / 0.3)',
              // Add pulsing animation for very active neurons
              animation: intensity > 0.7 ? 'pulse 1.5s ease-in-out infinite' : undefined,
            }}
            onClick={onClick}
          >
            {isDominant && (
              <Zap className="h-5 w-5 text-accent-foreground absolute -top-2 -right-2 animate-bounce" />
            )}
            {/* Enhanced activity indicator */}
            {isActive && (
              <div 
                className="absolute inset-1 rounded-full bg-gradient-to-br from-primary/60 to-primary animate-pulse" 
                style={{ 
                  opacity: 0.4 + intensity * 0.6,
                  transform: `scale(${0.3 + intensity * 0.7})`
                }}
              />
            )}
            {/* Core activation dot */}
            <div 
              className="absolute inset-3 rounded-full bg-foreground" 
              style={{ 
                opacity: intensity > 0.3 ? 0.8 : 0.2,
                transform: `scale(${intensity})`
              }}
            />
          </div>
        </TooltipTrigger>
        <TooltipContent className="max-w-48">
          <div className="text-xs space-y-1">
            <div className="font-semibold">Layer {layerIndex}, Neuron {neuronIndex}</div>
            <div>Activation: {activation.toFixed(4)}</div>
            <div>Intensity: {(intensity * 100).toFixed(1)}%</div>
            {isDominant && <div className="text-yellow-400">⚡ High Activity</div>}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

const AdaptiveConnections = ({ 
  from, 
  to, 
  fromActivations = [], 
  toActivations = [],
  showWeights = false 
}: { 
  from: number;
  to: number;
  fromActivations?: number[];
  toActivations?: number[];
  showWeights?: boolean;
}) => {
  const fromNodes = Math.min(from, MAX_NODES_TO_RENDER);
  const toNodes = Math.min(to, MAX_NODES_TO_RENDER);

  const fromY = Array.from({ length: fromNodes }, (_, i) => 
    fromNodes === 1 ? 50 : (i / (fromNodes - 1)) * 100
  );
  const toY = Array.from({ length: toNodes }, (_, i) => 
    toNodes === 1 ? 50 : (i / (toNodes - 1)) * 100
  );

  const connections = useMemo(() => {
    const lines = [];
    for (let i = 0; i < fromNodes; i++) {
      for (let j = 0; j < toNodes; j++) {
        const fromActivation = fromActivations[i] || 0;
        const toActivation = toActivations[j] || 0;
        const connectionStrength = (Math.abs(fromActivation) + Math.abs(toActivation)) / 2;
        
        // Enhanced visibility with higher base opacity
        const baseOpacity = 0.4; // Increased from 0.15
        const maxOpacity = 0.95; // Increased maximum
        const opacity = showWeights ? Math.max(baseOpacity, connectionStrength * maxOpacity) : baseOpacity;
        
        // Enhanced stroke width
        const minStrokeWidth = 1; // Increased from 0.5
        const strokeWidth = showWeights ? minStrokeWidth + connectionStrength * 2.5 : minStrokeWidth;
        
        const isActive = connectionStrength > 0.2; // Lower threshold
        const isVeryActive = connectionStrength > 0.6;
        
        // Enhanced color scheme using theme colors
        let strokeColor = 'hsl(var(--muted-foreground))'; // Default muted color
        if (isVeryActive) {
          strokeColor = 'hsl(var(--primary))'; // Primary theme color for very active
        } else if (isActive) {
          strokeColor = 'hsl(var(--primary) / 0.7)'; // Primary with opacity for active
        }
        
        lines.push(
          <line 
            key={`${i}-${j}`}
            x1="0%" 
            y1={`${fromY[i]}%`}
            x2="100%" 
            y2={`${toY[j]}%`}
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            opacity={opacity}
            className={cn(
              "transition-all duration-500 ease-in-out",
              isVeryActive && "connection-active"
            )}
            style={{
              filter: isVeryActive ? 'drop-shadow(0 0 4px hsl(var(--primary) / 0.8))' : 
                      isActive ? 'drop-shadow(0 0 2px hsl(var(--primary) / 0.6))' : undefined,
            }}
          />
        );
      }
    }
    return lines;
  }, [fromNodes, toNodes, fromActivations, toActivations, showWeights, fromY, toY]);

  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" fill="none">
      {connections}
    </svg>
  );
};

const LayerColumn = ({ 
  neuronCount, 
  activations, 
  layerStats,
  layerIndex,
  onNeuronClick 
}: { 
  neuronCount: number;
  activations: number[];
  layerStats?: LayerStats;
  layerIndex: number;
  onNeuronClick?: (layerIndex: number, neuronIndex: number) => void;
}) => {
  const nodesToRender = Math.min(neuronCount, MAX_NODES_TO_RENDER);
  const dominantNeurons = new Set(layerStats?.dominantFeatures.map(f => f.idx) || []);
  
  // Calculate average activity for the layer
  const avgActivity = activations.reduce((sum, act) => sum + Math.abs(act || 0), 0) / activations.length;
  const isLayerActive = avgActivity > 0.1;
  
  return (
    <div className={cn(
      "flex flex-col items-center justify-center h-full flex-shrink-0 w-12 z-10 relative gap-2",
      "transition-all duration-500",
      isLayerActive && "scale-105"
    )}>
      {Array.from({ length: nodesToRender }).map((_, i) => (
        <Neuron 
          key={i} 
          activation={activations[i] ?? 0}
          isDominant={dominantNeurons.has(i)}
          neuronIndex={i}
          layerIndex={layerIndex}
          onClick={() => onNeuronClick?.(layerIndex, i)}
        />
      ))}
      {neuronCount > MAX_NODES_TO_RENDER && (
        <div className={cn(
          "text-slate-300 text-3xl font-bold -my-1 flex items-center justify-center transition-all duration-300",
          isLayerActive && "text-blue-300 animate-pulse"
        )}>
          ⋮
        </div>
      )}
      
      {/* Layer activity indicator */}
      {isLayerActive && (
        <div className="absolute -bottom-2 w-full h-1 bg-gradient-to-r from-transparent via-primary to-transparent opacity-60 rounded-full animate-pulse" />
      )}
    </div>
  );
};

const LayerInfo = ({ 
  title, 
  neuronCount, 
  layerStats,
  onRemove, 
  onNeuronCountChange, 
  isRemovable, 
  isEditable 
}: { 
  title: string;
  neuronCount: number;
  layerStats?: LayerStats;
  onRemove?: () => void;
  onNeuronCountChange?: (neurons: number) => void;
  isRemovable: boolean;
  isEditable: boolean;
}) => (
  <div className="flex flex-col items-center text-center gap-2 w-28 min-w-0">
    <div className="font-semibold text-sm flex items-center gap-2 h-8 min-w-0">
      <span className="truncate">{title}</span>
      {isRemovable && onRemove && (
        <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground hover:text-destructive flex-shrink-0" onClick={onRemove}>
          <X className="w-3 h-3" />
        </Button>
      )}
    </div>
    
    {isEditable ? (
      <Input 
        type="number" 
        value={neuronCount} 
        onChange={(e) => onNeuronCountChange?.(Number(e.target.value))}
        className="w-20 h-8 text-center bg-input border-border text-foreground text-xs hover:border-primary" 
      />
    ) : (
      <div className="text-xs text-muted-foreground h-8 flex items-center px-2 bg-muted/50 rounded border border-border">
        {neuronCount} neurons
      </div>
    )}

    {layerStats && (
      <div className="flex flex-col gap-1 mt-1 w-full">
        <Badge 
          variant={layerStats.activationRate > 0.3 ? "default" : "secondary"} 
          className="text-xs h-5 w-full justify-center"
        >
          <Activity className="w-3 h-3 mr-1" />
          {(layerStats.activationRate * 100).toFixed(0)}%
        </Badge>
        
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="outline" className="text-xs h-5 cursor-help w-full justify-center">
                <Info className="w-3 h-3 mr-1" />
                Stats
              </Badge>
            </TooltipTrigger>
            <TooltipContent className="max-w-48">
              <div className="text-xs space-y-1">
                <div>Active: {layerStats.activeNeurons}/{layerStats.totalNeurons}</div>
                <div>Avg Activation: {layerStats.avgActivation.toFixed(3)}</div>
                <div>Max Activation: {layerStats.maxActivation.toFixed(3)}</div>
                <div>Top Features: {layerStats.dominantFeatures.slice(0, 3).map(f => f.idx).join(', ')}</div>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    )}
  </div>
);

export default function ModelVisualization({ 
  layersConfig, 
  activations, 
  layerStats = [],
  onAddLayer, 
  onRemoveLayer, 
  onNeuronCountChange 
}: ModelVisualizationProps) {
  const [selectedNeuron, setSelectedNeuron] = useState<{layer: number, neuron: number} | null>(null);
  const [showWeights, setShowWeights] = useState(false);
  
  const allLayers = [{ neurons: 784 }, ...layersConfig, { neurons: 10 }];
  const layerNames = ['Input', ...Array.from({length: layersConfig.length}, (_, i) => `Hidden ${i+1}`), 'Output'];

  const handleNeuronClick = (layerIndex: number, neuronIndex: number) => {
    setSelectedNeuron({layer: layerIndex, neuron: neuronIndex});
  };

  return (
    <div className="relative w-full h-full flex flex-col bg-card border border-border p-8 rounded-xl gap-8 shadow-2xl">
      {/* Controls */}
      <div className='flex-none flex items-center justify-between'>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onAddLayer} disabled={layersConfig.length >= 4}
                  className="border-border hover:border-primary hover:bg-primary/10">
            <Plus className="w-4 h-4 mr-2"/>
            Add Layer
          </Button>
          
          <Button 
            variant={showWeights ? "default" : "outline"} 
            size="sm" 
            onClick={() => setShowWeights(!showWeights)}
            className={showWeights ? "bg-primary hover:bg-primary/90" : "border-border hover:border-primary hover:bg-primary/10"}
          >
            <Eye className="w-4 h-4 mr-2"/>
            {showWeights ? 'Hide' : 'Show'} Weights
          </Button>
        </div>
        
        {selectedNeuron && (
          <Badge variant="secondary" className="flex items-center gap-1 bg-primary/20 text-primary border-primary/30">
            <Zap className="w-3 h-3" />
            Layer {selectedNeuron.layer}, Neuron {selectedNeuron.neuron}
          </Badge>
        )}
      </div>
      
      {/* Neural Network Visualization */}
      <div className="flex-grow flex justify-between items-center px-8 py-8 min-h-0 bg-background rounded-lg border border-border" style={{ minHeight: '500px' }}>
        {allLayers.map((layer, i) => (
          <React.Fragment key={i}>
            <LayerColumn 
              neuronCount={layer.neurons} 
              activations={activations[i] || []} 
              layerStats={layerStats[i]}
              layerIndex={i}
              onNeuronClick={handleNeuronClick}
            />
            {i < allLayers.length - 1 && (
              <div className="relative flex-grow h-full mx-6 flex items-center">
                <AdaptiveConnections 
                  from={allLayers[i].neurons} 
                  to={allLayers[i+1].neurons}
                  fromActivations={activations[i] || []}
                  toActivations={activations[i+1] || []}
                  showWeights={showWeights}
                />
                {/* Enhanced Activation Function Labels */}
                {(i > 0 && i < allLayers.length - 2) && (
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 font-mono text-sm text-primary-foreground bg-primary/90 border border-primary px-4 py-2 rounded-lg z-10 shadow-lg backdrop-blur-sm animate-pulse">
                    ReLU
                  </div>
                )}
                {i === allLayers.length - 2 && (
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 font-mono text-sm text-primary-foreground bg-accent border border-accent px-4 py-2 rounded-lg z-10 shadow-lg backdrop-blur-sm animate-pulse">
                    Softmax
                  </div>
                )}
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Layer Information */}
      <div className="flex-none flex justify-between items-start px-6 gap-4">
        {allLayers.map((layer, i) => (
          <LayerInfo 
            key={i}
            title={layerNames[i]}
            neuronCount={layer.neurons}
            layerStats={layerStats[i]}
            onRemove={i > 0 && i < allLayers.length - 1 ? () => onRemoveLayer(i - 1) : undefined}
            onNeuronCountChange={i > 0 && i < allLayers.length - 1 ? (neurons) => onNeuronCountChange(i - 1, neurons) : undefined}
            isRemovable={i > 0 && i < allLayers.length - 1}
            isEditable={i > 0 && i < allLayers.length - 1}
          />
        ))}
      </div>
      
      {/* Real-time Activity Summary */}
      {layerStats.length > 0 && (
        <div className="flex-none border-t border-border pt-3 bg-card">
          <div className="text-sm font-medium mb-2 flex items-center gap-2 text-foreground">
            <Activity className="w-4 h-4" />
            Network Activity Summary
          </div>
          <div className="flex gap-4 text-xs text-muted-foreground">
            <div>
              Total Active Neurons: {layerStats.reduce((sum, stats) => sum + stats.activeNeurons, 0)}
            </div>
            <div>
              Avg Network Activity: {(layerStats.reduce((sum, stats) => sum + stats.activationRate, 0) / layerStats.length * 100).toFixed(1)}%
            </div>
            <div>
              Most Active Layer: {layerNames[layerStats.reduce((maxIdx, stats, idx) => 
                stats.activationRate > layerStats[maxIdx].activationRate ? idx : maxIdx, 0
              )]}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

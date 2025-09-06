
'use client';

import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';

const PIXEL_DIMENSION = 28;
const LINE_WIDTH_PERCENT = 0.08; // Slightly thicker for better recognition
const BRUSH_SOFTNESS = 0.7; // Softer brush for more natural drawing

interface InferenceCanvasProps {
  onDrawEnd: (imageData: Float32Array) => void;
  onClear: () => void;
  isTraining: boolean;
}

export interface InferenceCanvasRef {
  clearCanvas: () => void;
  getImageData: () => Float32Array;
}

const InferenceCanvas = forwardRef<InferenceCanvasRef, InferenceCanvasProps>(({ onDrawEnd, onClear, isTraining }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const debounceTimeout = useRef<NodeJS.Timeout | null>(null);

  const drawGrid = (ctx: CanvasRenderingContext2D) => {
    const canvas = ctx.canvas;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'hsl(var(--border) / 0.1)';
    ctx.lineWidth = 1;

    for (let i = 0; i <= PIXEL_DIMENSION; i++) {
        const x = (i / PIXEL_DIMENSION) * canvas.width;
        const y = (i / PIXEL_DIMENSION) * canvas.height;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
  };

  const renderPixelData = (pixelData: Float32Array) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    drawGrid(ctx);

    const cellWidth = canvas.width / PIXEL_DIMENSION;
    const cellHeight = canvas.height / PIXEL_DIMENSION;

    for (let y = 0; y < PIXEL_DIMENSION; y++) {
      for (let x = 0; x < PIXEL_DIMENSION; x++) {
        const value = pixelData[y * PIXEL_DIMENSION + x];
        if (value > 0) {
          ctx.fillStyle = `rgba(255, 255, 255, ${value})`;
          ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
        }
      }
    }
  };

  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (canvas && container) {
      const { width, height } = container.getBoundingClientRect();
      const size = Math.min(width, height);
      canvas.width = size;
      canvas.height = size;
      clearCanvas(false);
    }
  };

  useEffect(() => {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, []);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if(canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        drawGrid(ctx);
      }
    }
  }, []);

  const getDownscaledImageData = (): Float32Array => {
    const canvas = canvasRef.current;
    if (!canvas) return new Float32Array(0);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = PIXEL_DIMENSION;
    tempCanvas.height = PIXEL_DIMENSION;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return new Float32Array(0);

    // Center the drawing with a bit of padding
    const boundingBox = getDrawingBoundingBox();
    const scale = 0.7; // Scale down to 70%
    const contentWidth = boundingBox.maxX - boundingBox.minX;
    const contentHeight = boundingBox.maxY - boundingBox.minY;
    
    if (contentWidth === 0 || contentHeight === 0) {
      return new Float32Array(PIXEL_DIMENSION * PIXEL_DIMENSION).fill(0);
    }

    const contentAspectRatio = contentWidth / contentHeight;
    const canvasAspectRatio = 1;
    
    let drawWidth, drawHeight;
    if (contentAspectRatio > canvasAspectRatio) {
        drawWidth = PIXEL_DIMENSION * scale;
        drawHeight = drawWidth / contentAspectRatio;
    } else {
        drawHeight = PIXEL_DIMENSION * scale;
        drawWidth = drawHeight * contentAspectRatio;
    }
    
    const drawX = (PIXEL_DIMENSION - drawWidth) / 2;
    const drawY = (PIXEL_DIMENSION - drawHeight) / 2;

    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0,0, PIXEL_DIMENSION, PIXEL_DIMENSION);
    tempCtx.drawImage(canvas, boundingBox.minX, boundingBox.minY, contentWidth, contentHeight, drawX, drawY, drawWidth, drawHeight);

    const imageData = tempCtx.getImageData(0, 0, PIXEL_DIMENSION, PIXEL_DIMENSION).data;
    const grayscaleData = new Float32Array(PIXEL_DIMENSION * PIXEL_DIMENSION);

    for (let i = 0; i < imageData.length; i += 4) {
      const grayscale = imageData[i] / 255.0; // Use R channel for grayscale
      grayscaleData[i / 4] = grayscale;
    }
    return grayscaleData;
  };
  
  const getDrawingBoundingBox = () => {
    const canvas = canvasRef.current;
    if(!canvas) return { minX: 0, minY: 0, maxX: 0, maxY: 0};
    const ctx = canvas.getContext('2d');
    if(!ctx) return { minX: 0, minY: 0, maxX: 0, maxY: 0};

    const pixels = ctx.getImageData(0,0,canvas.width, canvas.height).data;
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;

    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const alpha = pixels[(y * canvas.width + x) * 4 + 3];
            if (alpha > 0) {
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
        }
    }
    const padding = 20;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(canvas.width, maxX + padding);
    maxY = Math.min(canvas.height, maxY + padding);
    
    return { minX, minY, maxX, maxY };
  }

  const handleDrawEnd = () => {
    if (debounceTimeout.current) {
      clearTimeout(debounceTimeout.current);
    }
    debounceTimeout.current = setTimeout(() => {
      const imageData = getDownscaledImageData();
      renderPixelData(imageData);
      onDrawEnd(imageData);
    }, 200); // Faster response for better UX
  };

  // Enhanced drawing with pressure-like sensitivity
  const drawWithPressure = (ctx: CanvasRenderingContext2D, x: number, y: number, pressure = 1.0) => {
    const lineWidth = (ctx.canvas.width * LINE_WIDTH_PERCENT) * pressure;
    
    // Create gradient for softer brush
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, lineWidth / 2);
    gradient.addColorStop(0, `rgba(255, 255, 255, ${BRUSH_SOFTNESS * pressure})`);
    gradient.addColorStop(0.7, `rgba(255, 255, 255, ${BRUSH_SOFTNESS * pressure * 0.5})`);
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas || isTraining) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear previous drawing to start fresh
    drawGrid(ctx);

    setIsDrawing(true);
    const pos = getMousePos(canvas, e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (!isDrawing || isTraining) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const pos = getMousePos(canvas, e);
    
    // Enhanced drawing with adaptive pressure based on movement speed
    const now = Date.now();
    const timeDelta = now - ((draw as any).lastTime || now);
    const lastPos = (draw as any).lastPos || pos;
    const distance = Math.sqrt(Math.pow(pos.x - lastPos.x, 2) + Math.pow(pos.y - lastPos.y, 2));
    const speed = timeDelta > 0 ? distance / timeDelta : 0;
    
    // Simulate pressure based on speed (slower = more pressure)
    const pressure = Math.max(0.4, Math.min(1.0, 1.2 - speed * 0.05));
    
    drawWithPressure(ctx, pos.x, pos.y, pressure);
    
    (draw as any).lastTime = now;
    (draw as any).lastPos = pos;
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (ctx) {
        ctx.closePath();
    }
    handleDrawEnd();
  };
  
  const getMousePos = (canvas: HTMLCanvasElement, e: React.MouseEvent | React.TouchEvent) => {
    const rect = canvas.getBoundingClientRect();
    const touch = 'touches' in e ? e.touches[0] : null;
    const clientX = touch ? touch.clientX : (e as React.MouseEvent).clientX;
    const clientY = touch ? touch.clientY : (e as React.MouseEvent).clientY;

    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  };

  const clearCanvas = (propagate = true) => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        drawGrid(ctx);
        if(propagate) onClear();
      }
    }
  };
  
  useImperativeHandle(ref, () => ({
    clearCanvas,
    getImageData: getDownscaledImageData,
  }));

  return (
    <div ref={containerRef} className="w-full h-full flex items-center justify-center">
        <canvas
          ref={canvasRef}
          className="rounded-lg touch-none bg-background"
          style={{ imageRendering: 'pixelated' }}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
    </div>
  );
});

InferenceCanvas.displayName = 'InferenceCanvas';
export default InferenceCanvas;

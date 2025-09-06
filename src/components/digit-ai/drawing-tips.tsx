'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Lightbulb, Target, Zap } from 'lucide-react';

interface DrawingTipsProps {
  className?: string;
}

export default function DrawingTips({ className }: DrawingTipsProps) {
  const tips = [
    {
      category: "General",
      icon: <Lightbulb className="w-4 h-4" />,
      color: "bg-blue-500",
      tips: [
        "Draw digits large and centered",
        "Use bold, clear strokes",
        "Make sure the digit is complete",
        "Avoid extra marks or noise"
      ]
    },
    {
      category: "Accuracy",
      icon: <Target className="w-4 h-4" />,
      color: "bg-green-500", 
      tips: [
        "Connect all strokes properly",
        "Maintain consistent thickness",
        "Keep proportions realistic",
        "Clear the canvas between digits"
      ]
    },
    {
      category: "Recognition",
      icon: <Zap className="w-4 h-4" />,
      color: "bg-purple-500",
      tips: [
        "Follow standard digit shapes",
        "Avoid ambiguous forms",
        "Make distinctive features clear",
        "Use smooth, confident strokes"
      ]
    }
  ];

  const digitSpecificTips = {
    "0": "Make the oval shape clear and closed",
    "1": "Draw a straight vertical line",
    "2": "Connect curves and horizontal lines clearly",
    "3": "Ensure both curves are visible",
    "4": "Make crossing lines distinct",
    "5": "Draw top horizontal and bottom curve clearly",
    "6": "Close the bottom loop completely",
    "7": "Use a clear diagonal stroke",
    "8": "Make both loops clear and separated",
    "9": "Define the top loop and bottom stroke"
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Lightbulb className="w-4 h-4" />
          Drawing Tips for Better Recognition
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* General Tips */}
        {tips.map((category, index) => (
          <div key={index} className="space-y-2">
            <Badge variant="outline" className="text-xs">
              {category.icon}
              <span className="ml-1">{category.category}</span>
            </Badge>
            <ul className="text-xs space-y-1 ml-2">
              {category.tips.map((tip, tipIndex) => (
                <li key={tipIndex} className="flex items-start gap-1">
                  <span className="text-muted-foreground">•</span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}

        {/* Digit Specific Tips */}
        <div className="space-y-2">
          <Badge variant="outline" className="text-xs">
            <span className="text-lg mr-1">🔢</span>
            Digit-Specific
          </Badge>
          <div className="grid grid-cols-2 gap-1 text-xs">
            {Object.entries(digitSpecificTips).map(([digit, tip]) => (
              <div key={digit} className="flex items-start gap-1">
                <Badge variant="secondary" className="text-xs h-5 w-5 flex items-center justify-center p-0">
                  {digit}
                </Badge>
                <span className="text-muted-foreground text-[10px] leading-tight">{tip}</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

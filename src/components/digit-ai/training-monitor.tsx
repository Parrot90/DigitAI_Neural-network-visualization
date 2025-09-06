'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrainingHistory {
  epoch: number;
  loss: number;
  val_loss: number;
  acc?: number;
  val_acc?: number;
}

interface TrainingMonitorProps {
  trainingHistory: TrainingHistory[];
}

export default function TrainingMonitor({ trainingHistory }: TrainingMonitorProps) {
  const hasData = trainingHistory.length > 0;

  return (
    <div className="w-full h-full p-4 bg-card-foreground/5 rounded-lg">
      <ResponsiveContainer width="100%" height="100%">
      {hasData ? (
        <LineChart data={trainingHistory} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" />
          <XAxis dataKey="epoch" name="Epoch" style={{ fontSize: '0.75rem', fill: 'hsl(var(--muted-foreground))' }} />
          <YAxis domain={['dataMin', 'dataMax']} name="Loss" style={{ fontSize: '0.75rem', fill: 'hsl(var(--muted-foreground))' }} />
          <Tooltip
              contentStyle={{
                  background: 'hsl(var(--background))',
                  borderColor: 'hsl(var(--border))',
                  borderRadius: 'var(--radius)',
              }}
          />
          <Legend 
            wrapperStyle={{fontSize: "0.75rem", paddingTop: '10px'}} 
            payload={[
              { value: 'Train Loss', type: 'line', id: 'loss', color: 'hsl(var(--chart-1))' },
              { value: 'Val Loss', type: 'line', id: 'val_loss', color: 'hsl(var(--chart-2))' },
            ]}
          />
          <Line type="monotone" dataKey="loss" name="Train Loss" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
        </LineChart>
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground">
              Start training to see progress.
          </div>
      )}
      </ResponsiveContainer>
    </div>
  );
}
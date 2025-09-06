'use client';

import { zodResolver } from "@hookform/resolvers/zod";
import { Loader, Play, Zap, Settings, BarChart3, Brain, Download, Upload } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormDescription } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type { UseDigitAi } from "@/hooks/use-digit-ai";

const layerSchema = z.object({
  neurons: z.coerce.number().min(1, "At least 1 neuron is required.").max(1024, "Maximum 1024 neurons."),
});

export const configSchema = z.object({
  epochs: z.coerce.number().min(1).max(100),
  batchSize: z.coerce.number().min(1).max(512),
  trainingSamples: z.coerce.number().min(100).max(60000),
  learningRate: z.coerce.number().min(0.0001).max(1),
  optimizer: z.enum(["adam", "sgd", "rmsprop"]),
  layers: z.array(layerSchema).min(1, "At least one hidden layer is required."),
});

export type ConfigSchema = z.infer<typeof configSchema>;

type ConfigPanelProps = {
  useDigitAiManager: UseDigitAi;
};

export default function ConfigPanel({ useDigitAiManager }: ConfigPanelProps) {
  const {
    config,
    isTraining,
    isSuggesting,
    setConfig,
    handleTrain,
    trainOptimizedModel,
    trainingHistory,
    evaluateModel,
    suggestHyperparameters,
    saveModel,
    loadModel
  } = useDigitAiManager;

  const form = useForm<ConfigSchema>({
    resolver: zodResolver(configSchema),
    defaultValues: config,
    mode: 'onChange'
  });

  const onSubmit = (values: ConfigSchema) => {
    setConfig(values);
    handleTrain(values);
  };
  
  const finalAccuracy = trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].val_acc : null;

  return (
    <TooltipProvider>
      <Form {...form}>
        <div className="space-y-4">
          {/* Quick Actions Row */}
          <div className="flex flex-wrap items-center gap-2 p-3 bg-background/50 rounded-lg border">
            <div className="flex gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    type="button" 
                    variant="outline" 
                    className="h-8 text-xs" 
                    disabled={isTraining}
                    onClick={trainOptimizedModel}
                  >
                    <Brain className="w-3 h-3 mr-1" />
                    Robust Train
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Train with data augmentation for robust handwriting recognition</p>
                </TooltipContent>
              </Tooltip>
              
              <Button 
                type="submit" 
                className="h-8 text-xs" 
                disabled={isTraining}
                onClick={form.handleSubmit(onSubmit)}
              >
                {isTraining ? <Loader className="animate-spin w-3 h-3" /> : <Play className="w-3 h-3" />}
                Custom Train
              </Button>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    type="button" 
                    variant="outline" 
                    className="h-8 text-xs" 
                    disabled={isTraining}
                    onClick={evaluateModel}
                  >
                    <BarChart3 className="w-3 h-3 mr-1" />
                    Evaluate
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Evaluate current model on test dataset</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    type="button" 
                    variant={isSuggesting ? "secondary" : "outline"}
                    className="h-8 text-xs" 
                    disabled={isTraining || isSuggesting}
                    onClick={suggestHyperparameters}
                  >
                    {isSuggesting ? (
                      <Loader className="w-3 h-3 mr-1 animate-spin" />
                    ) : (
                      <Brain className="w-3 h-3 mr-1" />
                    )}
                    {isSuggesting ? 'AI Thinking...' : 'AI Suggest'}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{isSuggesting ? 'Getting AI-powered suggestions...' : 'Get AI-powered hyperparameter suggestions based on your network architecture'}</p>
                </TooltipContent>
              </Tooltip>

              {/* Model Save/Load Buttons */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    type="button" 
                    variant="outline"
                    className="h-8 text-xs" 
                    disabled={!finalAccuracy || isTraining}
                    onClick={saveModel}
                  >
                    <Download className="w-3 h-3 mr-1" />
                    Save
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Save trained model to your device</p>
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <label>
                    <Button 
                      type="button" 
                      variant="outline"
                      className="h-8 text-xs cursor-pointer" 
                      disabled={isTraining}
                      asChild
                    >
                      <span>
                        <Upload className="w-3 h-3 mr-1" />
                        Load
                      </span>
                    </Button>
                    <input
                      type="file"
                      accept=".json,.bin"
                      multiple
                      className="hidden"
                      onChange={async (e) => {
                        const files = e.target.files;
                        if (files && files.length > 0) {
                          try {
                            await loadModel(files);
                          } catch (error) {
                            console.error('Error loading model:', error);
                          }
                        }
                        // Reset the input so the same file can be selected again
                        e.target.value = '';
                      }}
                    />
                  </label>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Load a previously saved model (.json and .bin files)</p>
                </TooltipContent>
              </Tooltip>
            </div>

            <div className="flex-grow" />
            
            {finalAccuracy && (
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs">
                  Accuracy: {(finalAccuracy * 100).toFixed(1)}%
                </Badge>
                
                {/* Performance-based AI suggestion indicator */}
                {finalAccuracy < 0.9 && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button 
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-xs text-amber-600 hover:text-amber-800"
                        onClick={suggestHyperparameters}
                        disabled={isSuggesting}
                      >
                        <Brain className="w-3 h-3 mr-1" />
                        Boost Accuracy
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>AI can suggest improvements for better accuracy (target: 95%+)</p>
                    </TooltipContent>
                  </Tooltip>
                )}
                
                {finalAccuracy >= 0.95 && (
                  <Badge variant="default" className="text-xs bg-green-600">
                    Excellent!
                  </Badge>
                )}
              </div>
            )}
          </div>

          {/* Collapsible Advanced Settings */}
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between h-8 text-xs">
                <span className="flex items-center gap-2">
                  <Settings className="w-3 h-3" />
                  Advanced Hyperparameters
                </span>
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-2">
              <form className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 p-3 bg-background/30 rounded-lg border">
                <FormField control={form.control} name="epochs" render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">Epochs</FormLabel>
                    <FormControl>
                      <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                        <SelectTrigger className="w-full h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {[5, 10, 15, 20, 30, 50].map(e => 
                            <SelectItem key={e} value={String(e)}>{e}</SelectItem>
                          )}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormDescription className="text-xs">Training iterations</FormDescription>
                  </FormItem>
                )} />

                <FormField control={form.control} name="batchSize" render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">Batch Size</FormLabel>
                    <FormControl>
                      <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                        <SelectTrigger className="w-full h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {[16, 32, 64, 128, 256].map(b => 
                            <SelectItem key={b} value={String(b)}>{b}</SelectItem>
                          )}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormDescription className="text-xs">Samples per batch</FormDescription>
                  </FormItem>
                )} />

                <FormField control={form.control} name="trainingSamples" render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">Training Size</FormLabel>
                    <FormControl>
                      <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                        <SelectTrigger className="w-full h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {[5000, 10000, 20000, 30000, 40000, 50000, 60000].map(s => 
                            <SelectItem key={s} value={String(s)}>
                              {s.toLocaleString()} {s === 60000 ? '🔥 Full Dataset' : s >= 40000 ? '⭐ Recommended' : ''}
                            </SelectItem>
                          )}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormDescription className="text-xs">
                      Dataset size (60K = Full MNIST for best accuracy)
                    </FormDescription>
                  </FormItem>
                )} />

                <FormField control={form.control} name="learningRate" render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">Learning Rate</FormLabel>
                    <FormControl>
                      <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                        <SelectTrigger className="w-full h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {[0.0001, 0.0005, 0.001, 0.005, 0.01].map(lr => 
                            <SelectItem key={lr} value={String(lr)}>{lr}</SelectItem>
                          )}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormDescription className="text-xs">Step size</FormDescription>
                  </FormItem>
                )} />

                <FormField control={form.control} name="optimizer" render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">Optimizer</FormLabel>
                    <FormControl>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <SelectTrigger className="w-full h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="adam">Adam</SelectItem>
                          <SelectItem value="sgd">SGD</SelectItem>
                          <SelectItem value="rmsprop">RMSprop</SelectItem>
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormDescription className="text-xs">Optimization algorithm</FormDescription>
                  </FormItem>
                )} />
              </form>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </Form>
    </TooltipProvider>
  );
}

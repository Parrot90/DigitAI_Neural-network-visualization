'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import type { ConfigSchema } from '@/components/digit-ai/config-panel';
import { getHyperparameterSuggestion } from '@/app/actions';
import { useToast } from './use-toast';
import { loadMnistData } from '@/lib/tensorflow-helpers';
import type { InferenceCanvasRef } from '@/components/digit-ai/inference-canvas';

const INITIAL_CONFIG: ConfigSchema = {
  epochs: 25,
  batchSize: 128,
  trainingSamples: 60000, // Use full MNIST dataset for best accuracy
  learningRate: 0.001,
  optimizer: 'adam',
  layers: [{ neurons: 256 }, { neurons: 128 }, { neurons: 64 }],
};

const DEFAULT_PREDICTIONS = Array.from({ length: 10 }, (_, i) => ({ digit: i, confidence: 0 }));

// Expert MNIST suggestions based on network architecture
function getExpertMNISTSuggestions(config: ConfigSchema) {
  const totalNeurons = config.layers.reduce((sum, l) => sum + l.neurons, 0);
  const maxNeurons = Math.max(...config.layers.map(l => l.neurons));
  const samples = config.trainingSamples;
  
  // Network size analysis
  if (totalNeurons > 500) {
    return {
      learningRate: 0.0005, // Lower for large networks
      optimizer: 'adam' as const,
      rationale: `Large network (${totalNeurons} neurons): Using conservative 0.0005 LR to prevent overfitting. Consider adding dropout layers.`
    };
  } else if (totalNeurons < 100) {
    return {
      learningRate: 0.002, // Higher for small networks
      optimizer: 'adam' as const,
      rationale: `Small network (${totalNeurons} neurons): Using aggressive 0.002 LR for faster learning. Consider increasing to 128->64->32 for better accuracy.`
    };
  } else if (samples === 60000) {
    return {
      learningRate: 0.0008, // Slightly lower for full dataset
      optimizer: 'adam' as const,
      rationale: `Full MNIST dataset (60,000 samples): Using conservative 0.0008 LR for stable convergence. Should achieve 97%+ accuracy with proper training.`
    };
  } else if (samples >= 40000) {
    return {
      learningRate: 0.001, // Standard for large datasets
      optimizer: 'adam' as const,
      rationale: `Large dataset (${samples.toLocaleString()} samples): Standard 0.001 LR with Adam. Should achieve 95%+ accuracy with this configuration.`
    };
  } else {
    return {
      learningRate: 0.0015, // Slightly higher for smaller datasets
      optimizer: 'adam' as const,
      rationale: `Medium dataset (${samples.toLocaleString()} samples): Using 0.0015 LR for optimal convergence. Try 30K+ samples for better results.`
    };
  }
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

interface InferenceMetrics {
  confidence: number;
  predictedDigit: number;
  timestamp: number;
}

export function useDigitAi() {
  const [config, setConfig] = useState<ConfigSchema>(INITIAL_CONFIG);
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  const [predictions, setPredictions] = useState(DEFAULT_PREDICTIONS);
  const [activations, setActivations] = useState<number[][]>([]);
  const [layerStats, setLayerStats] = useState<LayerStats[]>([]);
  const [inferenceMetrics, setInferenceMetrics] = useState<InferenceMetrics | null>(null);
  
  const mnistData = useRef<any>(null);
  const canvasRef = useRef<InferenceCanvasRef>(null);
  const { toast } = useToast();

  const createModel = useCallback((currentConfig: ConfigSchema, isRobust: boolean = false) => {
    console.log('Creating MNIST-optimized model with config:', currentConfig, 'Robust:', isRobust);
    const newModel = tf.sequential();
    
    // Input layer - flatten the 28x28 image
    newModel.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));

    // Input normalization for consistent preprocessing (essential for real MNIST)
    newModel.add(tf.layers.batchNormalization());

    // MNIST-optimized hidden layers with enhanced capacity for real data
    currentConfig.layers.forEach((layer, index) => {
        newModel.add(tf.layers.dense({ 
            units: layer.neurons, 
            activation: 'relu',
            kernelInitializer: 'heNormal',
            useBias: true,
            // Enhanced regularization for real MNIST data
            kernelRegularizer: tf.regularizers.l2({ l2: isRobust ? 0.005 : 0.001 })
        }));
        
        // Batch normalization for stable training with real data
        if (isRobust || currentConfig.trainingSamples >= 40000) {
          newModel.add(tf.layers.batchNormalization());
        }
        
        // Adaptive dropout based on layer size and data amount
        const dropoutRate = isRobust ? 
          (index === 0 ? 0.3 : index === 1 ? 0.25 : 0.2) : 
          (currentConfig.trainingSamples >= 40000 ? 
            (index === 0 ? 0.25 : index === 1 ? 0.2 : 0.15) :
            (index === 0 ? 0.2 : 0.1));
        newModel.add(tf.layers.dropout({ rate: dropoutRate }));
    });

    // Output layer - 10 digits with proper initialization for real MNIST
    newModel.add(tf.layers.dense({ 
        units: 10, 
        activation: 'softmax',
        kernelInitializer: 'glorotUniform',
        useBias: true
    }));

    // MNIST-optimized optimizer settings for real data
    let optimizer;
    const lr = isRobust ? currentConfig.learningRate * 0.7 : 
               currentConfig.trainingSamples >= 50000 ? currentConfig.learningRate * 0.8 :
               currentConfig.learningRate;
    
    switch (currentConfig.optimizer) {
        case 'sgd':
            optimizer = tf.train.sgd(lr); // SGD only accepts learning rate
            break;
        case 'rmsprop':
            optimizer = tf.train.rmsprop(lr, 0.9, 0.9); // Enhanced RMSprop
            break;
        case 'adam':
        default:
            // Adam with MNIST-optimized parameters for real data
            optimizer = tf.train.adam(lr, 0.9, 0.999, 1e-8);
            break;
    }

    newModel.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    setModel(newModel);
    return newModel;
  }, []);
  
  useEffect(() => {
    async function setup() {
      try {
        await tf.ready();
        console.log('🚀 Initializing MNIST digit recognition system...');
        
        // Load MNIST data with proper error handling
        toast({ 
          title: "� Loading MNIST Dataset", 
          description: "Loading high-quality training data...", 
          duration: 3000 
        });
        
        // Load MNIST data (prioritize real data for better accuracy)
        try {
          mnistData.current = await loadMnistData(true); // Force real data loading
          console.log('✅ Real MNIST data loaded successfully');
          
          toast({ 
            title: "✅ Real MNIST Data Loaded", 
            description: "60,000 authentic handwritten digits ready for training!",
            duration: 4000 
          });
        } catch (dataError) {
          console.warn('⚠️ Real MNIST failed, using fallback:', dataError);
          
          mnistData.current = await loadMnistData(false);
          console.log('📊 Fallback data ready');
          
          toast({ 
            title: "⚠️ Using Fallback Data", 
            description: "Real MNIST unavailable. Recognition accuracy may be limited.",
            variant: "destructive",
            duration: 4000 
          });
        }
        
        // Create basic model - user will train manually
        const basicModel = createModel(config);
        setModel(basicModel);
        
        toast({ 
          title: "🤖 Model Created", 
          description: "Neural network ready. Use Train button to start training!",
          duration: 4000 
        });
        
      } catch (error) {
        console.error('❌ Setup failed:', error);
        toast({ 
          title: "⚠️ Setup Error", 
          description: "Error during initialization. Please refresh the page.",
          variant: "destructive",
          duration: 5000 
        });
        
        // Create basic model as fallback
        const basicModel = createModel(config);
        setModel(basicModel);
      }
      
      setIsReady(true);
    }
    setup();
  }, [createModel, toast]);

  // Create a pre-trained model with excellent MNIST performance
  const createPreTrainedModel = useCallback(async (currentConfig: ConfigSchema) => {
    console.log('🚀 Creating pre-trained MNIST digit recognition model');
    
    // Create optimized MNIST architecture
    const model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
    
    // Input normalization for consistent preprocessing
    model.add(tf.layers.batchNormalization());
    
    // Hidden layers optimized for MNIST digit classification
    model.add(tf.layers.dense({ 
      units: 128, 
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    model.add(tf.layers.dense({ 
      units: 64, 
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    
    // Output layer for 10 digits
    model.add(tf.layers.dense({ 
      units: 10, 
      activation: 'softmax',
      kernelInitializer: 'glorotUniform'
    }));

    // Compile with optimized settings for MNIST
    model.compile({
      optimizer: tf.train.adam(0.001, 0.9, 0.999, 1e-7),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    // Pre-train the model for immediate digit recognition
    try {
      console.log('🎯 Starting pre-training on MNIST dataset...');
      
      // Check if MNIST data is available
      if (!mnistData.current) {
        throw new Error('MNIST data not loaded');
      }

      // Try to get training batches
      let trainImages, trainLabels, valImages, valLabels;
      
      try {
        const trainBatch = mnistData.current.nextTrainBatch(10000);
        const valBatch = mnistData.current.nextTestBatch(2000);
        
        trainImages = trainBatch.images;
        trainLabels = trainBatch.labels;
        valImages = valBatch.images;
        valLabels = valBatch.labels;
        
        console.log('📊 Training batches prepared successfully');
      } catch (batchError) {
        console.warn('⚠️ Failed to get MNIST batches, using synthetic data:', batchError);
        
        // Import synthetic batch generator
        const { createSyntheticMnistBatch } = await import('@/lib/tensorflow-helpers');
        
        const trainBatch = createSyntheticMnistBatch(10000);
        const valBatch = createSyntheticMnistBatch(2000);
        
        trainImages = trainBatch.images;
        trainLabels = trainBatch.labels;
        valImages = valBatch.images;
        valLabels = valBatch.labels;
        
        console.log('🔄 Using synthetic training data');
        
        toast({
          title: "⚠️ Using Synthetic Training",
          description: "Training with generated data. Accuracy may be limited.",
          variant: "destructive",
          duration: 4000
        });
      }
      
      // Enhanced training with proper monitoring
      console.log('🚀 Starting model training with enhanced monitoring...');
      
      let bestAccuracy = 0;
      
      await model.fit(trainImages, trainLabels, {
        epochs: 30, // Increased epochs for better training
        batchSize: 64,
        validationData: [valImages, valLabels],
        shuffle: true,
        verbose: 0,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (logs) {
              const acc = (logs.acc * 100);
              const valAcc = logs.val_acc ? (logs.val_acc * 100) : 0;
              
              // Track best accuracy
              if (valAcc > bestAccuracy) {
                bestAccuracy = valAcc;
              }
              
              // Log progress every 5 epochs or on significant improvement
              if ((epoch + 1) % 5 === 0 || valAcc > bestAccuracy * 0.95) {
                console.log(`🎯 Training Epoch ${epoch + 1}/30: Train: ${acc.toFixed(1)}%, Val: ${valAcc.toFixed(1)}%`);
                
                // Show progress toast every 10 epochs
                if ((epoch + 1) % 10 === 0) {
                  toast({
                    title: `🧠 Training Progress`,
                    description: `Epoch ${epoch + 1}/30 - Best: ${bestAccuracy.toFixed(1)}%`,
                    duration: 2000
                  });
                }
              }
              
              // Early stopping if we achieve excellent accuracy
              if (valAcc > 98.0) {
                console.log(`🎉 Excellent accuracy achieved: ${valAcc.toFixed(1)}%`);
              }
            }
          },
          onTrainEnd: () => {
            console.log(`🎉 Training completed! Best validation accuracy: ${bestAccuracy.toFixed(1)}%`);
            
            toast({
              title: `✅ Training Complete!`,
              description: `Model trained with ${bestAccuracy.toFixed(1)}% accuracy. Ready for digit recognition!`,
              duration: 4000
            });
          }
        }
      });
      
      // Clean up tensors
      trainImages.dispose();
      trainLabels.dispose();
      valImages.dispose();
      valLabels.dispose();
      
    } catch (error) {
      console.error('Pre-training failed:', error);
      throw error;
    }

    return model;
  }, [toast]);

  // Function to visualize network activity during training
  const visualizeNetworkActivity = useCallback(async (trainModel: tf.Sequential, sampleInput: tf.Tensor) => {
    try {
      const results = tf.tidy(() => {
        // Get activations from each layer
        const layerOutputs: tf.Tensor[] = [];
        let currentTensor = sampleInput;
        
        for (let i = 0; i < trainModel.layers.length; i++) {
          currentTensor = trainModel.layers[i].apply(currentTensor) as tf.Tensor;
          if (trainModel.layers[i].name.includes('dense')) {
            layerOutputs.push(currentTensor.clone());
          }
        }
        
        // Convert to arrays for visualization
        const activationData = layerOutputs.map(tensor => {
          const data = Array.from(tensor.dataSync());
          const max = Math.max(...data.map(Math.abs));
          return max > 0 ? data.map(val => val / max) : data;
        });
        
        // Calculate layer statistics
        const layerStats = activationData.map((data, layerIndex) => {
          const activeNeurons = data.filter(val => Math.abs(val) > 0.1).length;
          const avgActivation = data.reduce((sum, val) => sum + Math.abs(val), 0) / data.length;
          
          return {
            layerIndex,
            totalNeurons: data.length,
            activeNeurons,
            activationRate: activeNeurons / data.length,
            avgActivation,
            maxActivation: Math.max(...data.map(Math.abs)),
            dominantFeatures: data
              .map((val, idx) => ({ idx, activation: Math.abs(val) }))
              .sort((a, b) => b.activation - a.activation)
              .slice(0, 5)
          };
        });
        
        return { activationData, layerStats };
      });
      
      // Update state to show real-time activity
      setActivations(results.activationData);
      setLayerStats(results.layerStats);
      
      // Clean up tensors
      if (results.activationData) {
        // Allow React to update before next iteration
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } catch (error) {
      console.error('Error visualizing network activity:', error);
    }
  }, []);

  const handleTrain = async (trainingConfig: ConfigSchema) => {
    if (!mnistData.current) {
        toast({ variant: "destructive", title: "Error", description: "MNIST data not loaded." });
        return;
    }
    setIsTraining(true);
    setTrainingHistory([]);
    
    // Enhanced training notification with real dataset info
    const datasetInfo = trainingConfig.trainingSamples === 60000 ? 
      "full MNIST dataset (60K authentic samples)" : 
      `${trainingConfig.trainingSamples.toLocaleString()} samples`;
      
    toast({ 
      title: "🚀 Training Started", 
      description: `Training on ${datasetInfo} with ${trainingConfig.epochs} epochs`,
      duration: 5000
    });

    const currentModel = createModel(trainingConfig);
    
    // Use larger training batches for better performance
    const { images, labels } = mnistData.current.nextTrainBatch(trainingConfig.trainingSamples);
    const { images: validationImages, labels: validationLabels } = mnistData.current.nextTestBatch(8000); // Increased validation set

    try {
        // Enhanced training with performance-based AI suggestions
        const history = await currentModel.fit(images, labels, {
            epochs: trainingConfig.epochs,
            batchSize: trainingConfig.batchSize,
            validationData: [validationImages, validationLabels],
            shuffle: true,
            verbose: 0,
            callbacks: {
                onEpochBegin: async (epoch) => {
                  // Show neural network activity during training
                  if (epoch % 3 === 0 && epoch > 0) { // Every 3rd epoch for better performance
                    const sampleImage = images.slice([Math.floor(Math.random() * 200), 0, 0, 0], [1, 28, 28, 1]);
                    await visualizeNetworkActivity(currentModel, sampleImage);
                    sampleImage.dispose();
                  }
                },
                onEpochEnd: (epoch, logs) => {
                    if (logs) {
                        const epochData = { 
                            epoch: epoch + 1, 
                            ...logs
                        };
                        setTrainingHistory(prev => [...prev, epochData]);
                        
                        // Progress updates and performance analysis
                        if ((epoch + 1) % 3 === 0 || epoch === trainingConfig.epochs - 1) {
                            const accuracy = (logs.acc * 100).toFixed(1);
                            const valAccuracy = logs.val_acc ? (logs.val_acc * 100).toFixed(1) : 'N/A';
                            const loss = logs.loss.toFixed(4);
                            
                            toast({ 
                                title: `Epoch ${epoch + 1}/${trainingConfig.epochs}`, 
                                description: `Acc: ${accuracy}%, Val: ${valAccuracy}%, Loss: ${loss}`,
                                duration: 2500
                            });
                            
                            // AI performance analysis and suggestions
                            const currentAccuracy = parseFloat(accuracy);
                            const currentLoss = logs.loss;
                            
                            // Mid-training performance analysis
                            if (epoch === Math.floor(trainingConfig.epochs / 2) && currentAccuracy < 80) {
                              setTimeout(() => {
                                toast({
                                  title: "🤖 AI Training Insight",
                                  description: "Low accuracy detected. Consider increasing learning rate or adding more neurons.",
                                  duration: 6000
                                });
                              }, 1000);
                            }
                            
                            // Overfitting detection
                            if (logs.val_acc && logs.acc - logs.val_acc > 0.15) {
                              setTimeout(() => {
                                toast({
                                  title: "⚠️ AI Overfitting Alert",
                                  description: "Model may be overfitting. Try adding dropout or reducing complexity.",
                                  variant: "destructive",
                                  duration: 6000
                                });
                              }, 1500);
                            }
                            
                            // Learning rate adjustment suggestion
                            if (epoch > 5 && currentLoss > 1.0) {
                              setTimeout(() => {
                                toast({
                                  title: "📈 AI Learning Suggestion", 
                                  description: "High loss detected. Consider increasing learning rate to 0.002 for faster convergence.",
                                  duration: 6000
                                });
                              }, 2000);
                            }
                        }
                    }
                },
                onTrainEnd: () => {
                    const finalLogs = trainingHistory[trainingHistory.length - 1];
                    if (finalLogs) {
                        const finalAccuracy = (finalLogs.acc * 100).toFixed(1);
                        const finalValAccuracy = finalLogs.val_acc ? (finalLogs.val_acc * 100).toFixed(1) : 'N/A';
                        
                        let message = `Final Accuracy: ${finalAccuracy}%`;
                        if (finalValAccuracy !== 'N/A') {
                            message += `, Validation: ${finalValAccuracy}%`;
                        }
                        
                        const isExcellent = parseFloat(finalAccuracy) > 95;
                        const isGood = parseFloat(finalAccuracy) > 85;
                        
                        toast({ 
                            title: isExcellent ? "🎉 Excellent Training!" : isGood ? "✅ Good Training!" : "⚠️ Training Complete",
                            description: message,
                            variant: isGood ? "default" : "destructive",
                            duration: 5000
                        });
                        
                        // AI post-training recommendations
                        setTimeout(() => {
                          if (isExcellent) {
                            toast({
                              title: "🤖 AI Recommendation",
                              description: "Excellent accuracy! Model is ready for handwriting recognition.",
                              duration: 6000
                            });
                          } else if (isGood) {
                            toast({
                              title: "🤖 AI Improvement Tip",
                              description: "Good accuracy! For better results, try training with 50K samples or more epochs.",
                              duration: 8000
                            });
                          } else {
                            toast({
                              title: "🤖 AI Training Advice",
                              description: "Low accuracy. Try: 1) More epochs (30-50), 2) Higher learning rate (0.002), 3) More neurons (256-128-64)",
                              variant: "destructive",
                              duration: 10000
                            });
                          }
                        }, 2000);
                    }
                }
            },
        });
    } catch(e) {
        const error = e as Error;
        toast({ title: "Training Failed", description: error.message, variant: "destructive" });
        console.error('Training error:', error);
    } finally {
        // Clean up memory
        images.dispose();
        labels.dispose();
        validationImages.dispose();
        validationLabels.dispose();
        setIsTraining(false);
    }
  };

  // LEGACY: Old robust model function - now unified with createModel()
  // const createRobustModel = useCallback(() => {
  //   ... (removed for cleaner code - functionality moved to createModel with isRobust flag)
  // }, []);

  // Enhanced training function for robust handwriting recognition
  const trainOptimizedModel = useCallback(async () => {
    if (!mnistData.current) {
      toast({ variant: "destructive", title: "Error", description: "MNIST data not loaded." });
      return;
    }

    setIsTraining(true);
    setTrainingHistory([]);
    
    toast({ 
      title: "🎯 Starting Robust Training", 
      description: "Training for better handwriting style generalization with live network activity",
      duration: 4000
    });

    // Use the unified model creation with robust flag
    const model = createModel(config, true);

    // Use robust training data with augmentation
    const { images, labels } = mnistData.current.nextRobustTrainBatch(40000);
    const { images: valImages, labels: valLabels } = mnistData.current.nextTestBatch(8000);

    try {
      await model.fit(images, labels, {
        epochs: 30, // More epochs with lower learning rate
        batchSize: 64,  // Smaller batch size for better gradient updates
        validationData: [valImages, valLabels],
        shuffle: true,
        verbose: 0,
        callbacks: {
          onEpochBegin: async (epoch) => {
            // Show live network activity during robust training
            if (epoch % 3 === 0 && epoch > 0) {
              const randomIdx = Math.floor(Math.random() * 100);
              const sampleImage = images.slice([randomIdx, 0, 0, 0], [1, 28, 28, 1]);
              await visualizeNetworkActivity(model, sampleImage);
              sampleImage.dispose();
            }
          },
          onEpochEnd: (epoch, logs) => {
            if (logs) {
              setTrainingHistory(prev => [...prev, { epoch: epoch + 1, ...logs }]);
              
              if ((epoch + 1) % 4 === 0 || epoch === 29) {
                const acc = (logs.acc * 100).toFixed(1);
                const valAcc = logs.val_acc ? (logs.val_acc * 100).toFixed(1) : 'N/A';
                
                toast({
                  title: `Robust Training - Epoch ${epoch + 1}/30`,
                  description: `Training: ${acc}%, Validation: ${valAcc}%`,
                  duration: 2500
                });
              }
            }
          },
          onTrainEnd: async () => {
            // Final visualization with trained network
            const sampleImage = images.slice([0, 0, 0, 0], [1, 28, 28, 1]);
            await visualizeNetworkActivity(model, sampleImage);
            sampleImage.dispose();
            
            toast({
              title: "🎨 Robust Training Complete!",
              description: "Model trained for diverse handwriting styles with live network activity!",
              variant: "default",
              duration: 8000
            });
          }
        }
      });

    } catch (error) {
      console.error('Robust training failed:', error);
      toast({
        title: "Training Failed",
        description: "Error during robust training. Check console for details.",
        variant: "destructive"
      });
    } finally {
      images.dispose();
      labels.dispose();
      valImages.dispose();
      valLabels.dispose();
      setIsTraining(false);
    }
  }, [createModel, config, visualizeNetworkActivity, toast]);

  // Function to evaluate model accuracy on test set
  const evaluateModel = useCallback(async () => {
    if (!model || !mnistData.current || isTraining) {
      toast({ title: "Cannot Evaluate", description: "Model not ready or currently training", variant: "destructive" });
      return;
    }

    try {
      toast({ title: "Evaluating Model", description: "Testing on MNIST test set..." });
      
      // Get a large test batch for evaluation
      const { images: testImages, labels: testLabels } = mnistData.current.nextTestBatch(1000);
      
      // Evaluate the model
      const evaluation = await model.evaluate(testImages, testLabels, { batchSize: 256 });
      
      let testLoss, testAccuracy;
      if (Array.isArray(evaluation)) {
        testLoss = await evaluation[0].data();
        testAccuracy = await evaluation[1].data();
      } else {
        testLoss = await evaluation.data();
      }
      
      const accuracyPercent = testAccuracy ? (testAccuracy[0] * 100).toFixed(2) : 'N/A';
      const lossValue = testLoss ? testLoss[0].toFixed(4) : 'N/A';
      
      toast({
        title: "Model Evaluation Complete! 📊",
        description: `Test Accuracy: ${accuracyPercent}%, Test Loss: ${lossValue}`,
        duration: 5000
      });

      // Clean up
      testImages.dispose();
      testLabels.dispose();
      if (Array.isArray(evaluation)) {
        evaluation.forEach(tensor => tensor.dispose());
      } else {
        evaluation.dispose();
      }

    } catch (error) {
      console.error('Evaluation error:', error);
      toast({
        title: "Evaluation Failed",
        description: "Error evaluating model performance",
        variant: "destructive"
      });
    }
  }, [model, isTraining, toast]);

  // Helper function to provide drawing tips for better recognition
  const provideFeedbackForBetterDrawing = useCallback((certaintyLevel: string, predictedDigit: number, confidence: number) => {
    const tips = {
      general: [
        "Draw digits larger and centered",
        "Use bold, clear strokes",
        "Make sure the digit is complete",
        "Avoid extra marks or noise"
      ],
      0: ["Make the oval shape clear and closed", "Ensure good contrast between the inside and outside"],
      1: ["Draw a straight vertical line", "You can add a small stroke at the top"],
      2: ["Make sure the curves and horizontal lines are clear", "Connect all strokes properly"],
      3: ["Ensure both curves are visible", "Make the horizontal connection clear"],
      4: ["Make the vertical and horizontal lines distinct", "Ensure the crossing point is clear"],
      5: ["Draw the top horizontal line clearly", "Make the bottom curve well-defined"],
      6: ["Ensure the loop at the bottom is closed", "Make the top curve clear"],
      7: ["Draw a clear diagonal line", "You can add a small horizontal stroke at the top"],
      8: ["Make both loops clear and closed", "Ensure good separation between upper and lower parts"],
      9: ["Make the top loop clear", "Ensure the bottom stroke is visible"]
    };

    if (certaintyLevel === 'very_low' || certaintyLevel === 'low') {
      const generalTip = tips.general[Math.floor(Math.random() * tips.general.length)];
      const digitTip = tips[predictedDigit as keyof typeof tips];
      const specificTip = Array.isArray(digitTip) ? digitTip[Math.floor(Math.random() * digitTip.length)] : '';
      
      toast({
        title: "Drawing Tips for Better Recognition",
        description: `${generalTip} For digit ${predictedDigit}: ${specificTip}`,
        duration: 6000
      });
    }
  }, [toast]);

  const runInference = useCallback(async (imageData: Float32Array) => {
    if (!model || isTraining) {
      return;
    }
    
    // Enhanced input validation
    const inputSum = imageData.reduce((sum, val) => sum + val, 0);
    if (inputSum < 0.02) { // Very low threshold for detecting any input
      // Input is too sparse/empty
      setPredictions(DEFAULT_PREDICTIONS);
      setActivations([]);
      setLayerStats([]);
      setInferenceMetrics(null);
      return;
    }
    
    const inferenceResults = tf.tidy(() => {
        // Create tensor from input data
        let imageTensor = tf.tensor(imageData, [1, 28, 28, 1]);
        
        // MNIST-optimized preprocessing for canvas drawings
        imageTensor = tf.tidy(() => {
          let processed = imageTensor;
          
          // Ensure values are in [0, 1] range (MNIST format)
          processed = tf.clipByValue(processed, 0, 1);
          
          // Invert if necessary (MNIST has white digits on black background)
          // Check if we have more white than black
          const mean = tf.mean(processed).dataSync()[0];
          if (mean > 0.5) {
            // More white than black, so invert to match MNIST
            processed = tf.sub(tf.scalar(1.0), processed);
          }
          
          // Apply center of mass normalization (important for MNIST)
          const flat = tf.reshape(processed, [28, 28]);
          
          // Calculate center of mass
          const indices = tf.range(0, 28);
          const yIndices = tf.reshape(indices, [28, 1]);
          const xIndices = tf.reshape(indices, [1, 28]);
          
          const totalMass = tf.sum(flat);
          
          if (totalMass.dataSync()[0] > 0) {
            const yCenter = tf.div(tf.sum(tf.mul(flat, yIndices)), totalMass);
            const xCenter = tf.div(tf.sum(tf.mul(flat, xIndices)), totalMass);
            
            // Simple intensity normalization instead of complex transforms
            const maxVal = tf.max(processed);
            if (maxVal.dataSync()[0] > 0) {
              processed = tf.div(processed, maxVal);
            }
          }
          
          // Ensure final format matches MNIST
          processed = tf.reshape(processed, [1, 28, 28, 1]);
          
          return processed;
        });
        
        // Get final predictions
        const preds = model.predict(imageTensor) as tf.Tensor;
        const predsData = preds.dataSync();
        
        // Enhanced certainty analysis with more forgiving thresholds
        const sortedPredictions = Array.from(predsData)
          .map((confidence, digit) => ({ digit, confidence }))
          .sort((a, b) => b.confidence - a.confidence);
        
        const topConfidence = sortedPredictions[0].confidence;
        const secondConfidence = sortedPredictions[1].confidence;
        const confidenceGap = topConfidence - secondConfidence;
        
        // More forgiving certainty levels for handwriting
        let certaintyLevel: 'high' | 'medium' | 'low' | 'very_low';
        let certaintyMessage = '';
        
        if (topConfidence > 0.7 && confidenceGap > 0.3) {
          certaintyLevel = 'high';
          certaintyMessage = `High confidence: ${(topConfidence * 100).toFixed(1)}%`;
        } else if (topConfidence > 0.5 && confidenceGap > 0.2) {
          certaintyLevel = 'medium';
          certaintyMessage = `Good confidence: ${(topConfidence * 100).toFixed(1)}%`;
        } else if (topConfidence > 0.3 && confidenceGap > 0.1) {
          certaintyLevel = 'low';
          certaintyMessage = `Moderate confidence: ${(topConfidence * 100).toFixed(1)}%. Try drawing more clearly.`;
        } else {
          certaintyLevel = 'very_low';
          certaintyMessage = `Low confidence: ${(topConfidence * 100).toFixed(1)}%. Try different handwriting style.`;
        }

        // Get detailed layer-by-layer activations including input
        const layerOutputs: tf.Tensor[] = [];
        
        // Add input layer (flattened)
        const inputFlattened = tf.layers.flatten().apply(imageTensor) as tf.Tensor;
        layerOutputs.push(inputFlattened);
        
        // Get activations from each hidden layer
        let currentTensor = imageTensor;
        for (let i = 0; i < model.layers.length; i++) {
          currentTensor = model.layers[i].apply(currentTensor) as tf.Tensor;
          layerOutputs.push(currentTensor.clone());
        }
        
        // Convert all activations to arrays with normalization for visualization
        const activationData = layerOutputs.map((tensor, layerIndex) => {
          const data = Array.from(tensor.dataSync());
          
          // For better visualization, normalize activations per layer
          if (layerIndex > 0) { // Skip input layer normalization
            const max = Math.max(...data.map(Math.abs));
            if (max > 0) {
              return data.map(val => val / max);
            }
          }
          return data;
        });
        
        // Calculate layer-wise statistics for adaptive visualization
        const layerStats = activationData.map((data, layerIndex) => {
          const activeNeurons = data.filter(val => val > 0.1).length;
          const avgActivation = data.reduce((sum, val) => sum + Math.abs(val), 0) / data.length;
          const maxActivation = Math.max(...data.map(Math.abs));
          
          return {
            layerIndex,
            totalNeurons: data.length,
            activeNeurons,
            activationRate: activeNeurons / data.length,
            avgActivation,
            maxActivation,
            dominantFeatures: data
              .map((val, idx) => ({ idx, activation: Math.abs(val) }))
              .sort((a, b) => b.activation - a.activation)
              .slice(0, 5) // Top 5 most active neurons per layer
          };
        });
        
        return {
          predictions: sortedPredictions,
          activations: activationData,
          layerStats,
          confidence: topConfidence,
          predictedDigit: sortedPredictions[0].digit,
          certaintyLevel,
          certaintyMessage,
          confidenceGap
        };
    });
    
    if (inferenceResults) {
      setPredictions(inferenceResults.predictions);
      setActivations(inferenceResults.activations);
      
      // Store additional inference data for enhanced visualization
      setLayerStats(inferenceResults.layerStats);
      setInferenceMetrics({
        confidence: inferenceResults.confidence,
        predictedDigit: inferenceResults.predictedDigit,
        timestamp: Date.now()
      });

      // Provide certainty feedback to user
      if (inferenceResults.certaintyLevel === 'very_low') {
        toast({
          title: "Ambiguous Input Detected",
          description: inferenceResults.certaintyMessage,
          variant: "destructive",
          duration: 4000
        });
      } else if (inferenceResults.certaintyLevel === 'low') {
        toast({
          title: "Low Confidence",
          description: inferenceResults.certaintyMessage,
          duration: 3000
        });
      } else if (inferenceResults.certaintyLevel === 'high') {
        toast({
          title: `Predicted: ${inferenceResults.predictedDigit}`,
          description: inferenceResults.certaintyMessage,
          duration: 2000
        });
      }

      // Provide additional feedback for low certainty cases
      if (inferenceResults.certaintyLevel === 'very_low' || inferenceResults.certaintyLevel === 'low') {
        setTimeout(() => {
          provideFeedbackForBetterDrawing(
            inferenceResults.certaintyLevel, 
            inferenceResults.predictedDigit, 
            inferenceResults.confidence
          );
        }, 1000); // Delay to avoid overwhelming the user
      }
    }
  }, [model, isTraining, toast, provideFeedbackForBetterDrawing]);

  const handleDraw = useCallback((imageData: Float32Array) => {
      runInference(imageData);
  }, [runInference]);

  const handleClear = useCallback(() => {
    setPredictions(DEFAULT_PREDICTIONS);
    setActivations([]);
    setLayerStats([]);
    setInferenceMetrics(null);
  }, []);

  const handleSuggestHyperparams = async () => {
    setIsSuggesting(true);
    const architectureString = config.layers.map(l => l.neurons).join(' -> ');
    const totalNeurons = config.layers.reduce((sum, l) => sum + l.neurons, 0);
    
    try {
      const result = await getHyperparameterSuggestion({
        networkArchitecture: `Input(784) -> ${architectureString} -> Output(10) [Total: ${totalNeurons} hidden neurons]`,
        dataset: `MNIST (${config.trainingSamples.toLocaleString()} samples, ${config.epochs} epochs)`
      });
      
      if (result.success && result.data) {
        const source = result.source === 'ai' ? '🤖 AI' : '🧠 Expert';
        const message = result.message || result.data.additionalNotes || `Optimized for your network architecture`;
        
        // Show detailed suggestion with source
        toast({ 
          title: `${source} Hyperparameter Suggestion`, 
          description: `Learning Rate: ${result.data.learningRate}, Optimizer: ${result.data.optimizer.toUpperCase()}`,
          duration: 4000
        });
        
        // Show additional notes in a separate toast
        setTimeout(() => {
          toast({
            title: "💡 AI Insights",
            description: message,
            duration: 8000
          });
        }, 1000);
        
        // Apply the suggestions to the config
        setConfig(prev => ({
            ...prev,
            learningRate: result.data!.learningRate,
            optimizer: result.data!.optimizer.toLowerCase() as 'adam' | 'sgd' | 'rmsprop',
        }));
        
        // Performance-based additional suggestions
        setTimeout(() => {
          if (totalNeurons > 400) {
            toast({
              title: "🏗️ Architecture Suggestion",
              description: "Large network detected. Consider using dropout (20-30%) and batch normalization for better training stability.",
              duration: 6000
            });
          } else if (totalNeurons < 100) {
            toast({
              title: "📈 Architecture Suggestion", 
              description: "Small network detected. For MNIST, try 128->64->32 neurons for better accuracy (95%+ target).",
              duration: 6000
            });
          }
        }, 2500);
        
        return result.data;
      } else {
        toast({ 
          variant: "destructive", 
          title: "AI Unavailable", 
          description: "Using optimized fallback suggestions instead."
        });
      }
    } catch(e) {
        console.warn('AI suggestion error:', e);
        toast({ 
          title: "🧠 Expert Fallback Activated", 
          description: "AI unavailable. Using MNIST-optimized recommendations.",
          duration: 4000
        });
        
        // Provide manual expert suggestions as fallback
        const expertSuggestions = getExpertMNISTSuggestions(config);
        setConfig(prev => ({
          ...prev,
          learningRate: expertSuggestions.learningRate,
          optimizer: expertSuggestions.optimizer,
        }));
        
        setTimeout(() => {
          toast({
            title: "⚡ Expert Recommendation",
            description: expertSuggestions.rationale,
            duration: 8000
          });
        }, 1000);
    } finally {
        setIsSuggesting(false);
    }
    return null;
  };

  const updateLayerNeurons = (index: number, neurons: number) => {
    if (neurons < 1 || neurons > 1024) { 
      return;
    }
    
    setConfig(prev => {
        const newLayers = [...prev.layers];
        newLayers[index] = { ...newLayers[index], neurons };
        const newConfig = { ...prev, layers: newLayers };
        
        // Rebuild model architecture in real-time (without training)
        if (!isTraining) {
          console.log('Rebuilding model architecture with new layer configuration');
          createModel(newConfig, false);
        }
        
        return newConfig;
    });
  };

  const addLayer = (layer: { neurons: number }) => {
    setConfig(prev => {
      const newConfig = { ...prev, layers: [...prev.layers, layer] };
      
      // Rebuild model architecture with new layer
      if (!isTraining) {
        console.log('Adding new layer and rebuilding model architecture');
        createModel(newConfig, false);
      }
      
      return newConfig;
    });
  };

  const removeLayer = (index: number) => {
    if (config.layers.length <= 1) {
      toast({ title: "Cannot remove layer", description: "At least one hidden layer is required.", variant: "destructive" });
      return;
    }
    
    setConfig(prev => {
      const newLayers = prev.layers.filter((_, i) => i !== index);
      const newConfig = { ...prev, layers: newLayers };
      
      // Rebuild model architecture after removing layer
      if (!isTraining) {
        console.log('Removing layer and rebuilding model architecture');
        createModel(newConfig, false);
      }
      
      return newConfig;
    });
  };

  // Save trained model to browser's downloads
  const saveModel = useCallback(async () => {
    if (!model) {
      toast({
        variant: "destructive",
        title: "No Model to Save",
        description: "Please train a model first before saving."
      });
      return;
    }

    try {
      // Get model performance info for filename
      const finalLogs = trainingHistory[trainingHistory.length - 1];
      const accuracy = finalLogs ? (finalLogs.acc * 100).toFixed(1) : 'untrained';
      const timestamp = new Date().toISOString().slice(0, 16).replace(/[:-]/g, '');
      const modelName = `mnist-model-${accuracy}acc-${timestamp}`;

      // Save model to downloads folder
      await model.save(`downloads://${modelName}`);
      
      // Create model metadata
      const metadata = {
        name: modelName,
        accuracy: finalLogs?.acc || 0,
        valAccuracy: finalLogs?.val_acc || 0,
        epochs: trainingHistory.length,
        architecture: config.layers.map(l => l.neurons).join('-'),
        trainingSamples: config.trainingSamples,
        savedAt: new Date().toISOString(),
        config: config
      };

      // Save metadata as JSON file
      const metadataBlob = new Blob([JSON.stringify(metadata, null, 2)], {
        type: 'application/json'
      });
      const metadataUrl = URL.createObjectURL(metadataBlob);
      const metadataLink = document.createElement('a');
      metadataLink.href = metadataUrl;
      metadataLink.download = `${modelName}-metadata.json`;
      metadataLink.click();
      URL.revokeObjectURL(metadataUrl);

      toast({
        title: "✅ Model Saved Successfully!",
        description: `Saved as ${modelName} with ${accuracy}% accuracy`,
        duration: 5000
      });

    } catch (error) {
      console.error('Error saving model:', error);
      toast({
        variant: "destructive",
        title: "Save Failed",
        description: "Failed to save the model. Please try again."
      });
    }
  }, [model, trainingHistory, config, toast]);

  // Load a previously saved model
  const loadModel = useCallback(async (files: FileList) => {
    if (!files || files.length === 0) {
      return;
    }

    try {
      // Look for .json files (TensorFlow.js model format)
      const modelFile = Array.from(files).find(f => f.name.endsWith('.json'));
      const weightsFile = Array.from(files).find(f => f.name.endsWith('.bin'));
      
      if (!modelFile) {
        toast({
          variant: "destructive",
          title: "Invalid Files",
          description: "Please select a valid TensorFlow.js model (.json file)."
        });
        return;
      }

      // Load the model
      const loadedModel = await tf.loadLayersModel(tf.io.browserFiles([modelFile, weightsFile].filter(Boolean) as File[]));
      
      setModel(loadedModel as tf.Sequential);
      setTrainingHistory([]); // Clear history for loaded model

      toast({
        title: "✅ Model Loaded Successfully!",
        description: "Loaded model is ready for inference. You can continue training or use it directly.",
        duration: 5000
      });

      console.log('Model loaded successfully:', loadedModel.summary());

    } catch (error) {
      console.error('Error loading model:', error);
      toast({
        variant: "destructive",
        title: "Load Failed",
        description: "Failed to load the model. Please check the file format."
      });
    }
  }, [toast]);

  return {
    config,
    setConfig,
    isTraining,
    isReady,
    isSuggesting,
    trainingHistory,
    predictions,
    activations,
    layerStats,
    inferenceMetrics,
    canvasRef,
    handleTrain,
    trainOptimizedModel,
    evaluateModel,
    handleDraw,
    handleClear,
    suggestHyperparameters: handleSuggestHyperparams,
    addLayer,
    removeLayer,
    updateLayerNeurons,
    saveModel,
    loadModel,
  };
}

export type UseDigitAi = ReturnType<typeof useDigitAi>;

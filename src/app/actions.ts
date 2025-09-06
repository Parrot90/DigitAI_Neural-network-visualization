'use server';

import { suggestHyperparameters } from '@/ai/flows/suggest-hyperparameters';
import type { SuggestHyperparametersInput } from '@/ai/flows/suggest-hyperparameters';

// Fallback AI suggestions based on MNIST best practices
function getFallbackSuggestions(input: SuggestHyperparametersInput) {
  const architecture = input.networkArchitecture.toLowerCase();
  const dataset = input.dataset.toLowerCase();
  
  // Parse network size from architecture string
  const hasLargeNetwork = architecture.includes('256') || architecture.includes('512');
  const hasSmallNetwork = architecture.includes('32') || architecture.includes('64');
  
  if (dataset.includes('mnist')) {
    if (hasLargeNetwork) {
      return {
        learningRate: 0.0005, // Lower LR for larger networks
        optimizer: 'adam',
        additionalNotes: 'Large network detected. Using lower learning rate (0.0005) to prevent overfitting. Consider adding dropout layers and using batch normalization for better stability.'
      };
    } else if (hasSmallNetwork) {
      return {
        learningRate: 0.002, // Higher LR for smaller networks
        optimizer: 'adam',
        additionalNotes: 'Small network detected. Using higher learning rate (0.002) for faster convergence. Consider increasing network size if accuracy is below 90%.'
      };
    } else {
      return {
        learningRate: 0.001, // Standard MNIST learning rate
        optimizer: 'adam',
        additionalNotes: 'Standard MNIST configuration. Adam optimizer with 0.001 learning rate typically achieves 95%+ accuracy. Train for 20-30 epochs with 30K+ samples for best results.'
      };
    }
  }
  
  // General fallback for other datasets
  return {
    learningRate: 0.001,
    optimizer: 'adam',
    additionalNotes: 'General recommendation: Adam optimizer with 0.001 learning rate. Adjust based on training performance - increase if loss decreases slowly, decrease if training is unstable.'
  };
}

export async function getHyperparameterSuggestion(input: SuggestHyperparametersInput) {
  try {
    // First try Genkit AI suggestions
    const result = await suggestHyperparameters(input);
    return { success: true, data: result, source: 'ai' };
  } catch (error) {
    console.warn('AI suggestions failed, using fallback:', error);
    
    // Fallback to rule-based suggestions
    const fallbackResult = getFallbackSuggestions(input);
    return { 
      success: true, 
      data: fallbackResult, 
      source: 'fallback',
      message: 'Using optimized fallback suggestions (AI unavailable)'
    };
  }
}

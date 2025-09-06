'use server';

/**
 * @fileOverview An AI agent that suggests hyperparameters for a neural network based on the architecture and dataset.
 *
 * - suggestHyperparameters - A function that suggests hyperparameters.
 * - SuggestHyperparametersInput - The input type for the suggestHyperparameters function.
 * - SuggestHyperparametersOutput - The return type for the suggestHyperparameters function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SuggestHyperparametersInputSchema = z.object({
  networkArchitecture: z
    .string()
    .describe('The architecture of the neural network (e.g., number of layers and neurons per layer).'),
  dataset: z.string().describe('The dataset being used for training (e.g., MNIST).'),
});
export type SuggestHyperparametersInput = z.infer<typeof SuggestHyperparametersInputSchema>;

const SuggestHyperparametersOutputSchema = z.object({
  learningRate: z.number().describe('The suggested learning rate for the neural network.'),
  optimizer: z.string().describe('The suggested optimizer setting (e.g., Adam, SGD).'),
  additionalNotes: z
    .string()
    .optional()
    .describe('Any additional notes or considerations for hyperparameter tuning.'),
});
export type SuggestHyperparametersOutput = z.infer<typeof SuggestHyperparametersOutputSchema>;

export async function suggestHyperparameters(
  input: SuggestHyperparametersInput
): Promise<SuggestHyperparametersOutput> {
  return suggestHyperparametersFlow(input);
}

const prompt = ai.definePrompt({
  name: 'suggestHyperparametersPrompt',
  input: {schema: SuggestHyperparametersInputSchema},
  output: {schema: SuggestHyperparametersOutputSchema},
  prompt: `You are an expert in deep learning hyperparameter tuning.

  Based on the provided network architecture and dataset, suggest a good starting point for the hyperparameters, including learning rate and optimizer settings.

  Network Architecture: {{{networkArchitecture}}}
  Dataset: {{{dataset}}}

  Consider the following:
  - Typical learning rate ranges for different optimizers.
  - The complexity of the network architecture.
  - Characteristics of the dataset.
  - Provide a brief explanation for your choices.

  Output the learning rate as a floating point number.
  Output the optimizer as a string.
`,
});

const suggestHyperparametersFlow = ai.defineFlow(
  {
    name: 'suggestHyperparametersFlow',
    inputSchema: SuggestHyperparametersInputSchema,
    outputSchema: SuggestHyperparametersOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);

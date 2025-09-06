# **App Name**: DigitAI

## Core Features:

- Inference Canvas: Implement an interactive pixelated 28x28 canvas that receives a user-drawn digit for inference. A 'Clear' button resets the canvas.
- Prediction Chart: Dynamically render a horizontal bar chart that visualizes confidence scores for each digit (0-9) after model inference. The digit with the highest confidence is highlighted for quick identification.
- Model Visualization: Visually represent the neural network architecture including an input layer with 784 neurons, two hidden layers (256 and 128 neurons respectively) with ReLU activation, and an output layer (10 neurons) with softmax activation.
- Training Configuration: Create a configuration panel for setting training parameters: number of epochs, batch size, and training sample size. Also provide a 'Train Model' button to initiate the training process.
- Model Trainer: Train a model using the MNIST dataset. Pixel values must be normalized to the range [0, 1]. Split the data into training and validation sets.
- Training Monitor: Present real-time training and validation loss via a line chart updated every epoch. Display final training accuracy prominently post-training. Provide controls to optionally animate neuron activations using the neural network visualization tool.
- Browser Execution: Run it in the browser using Tensorflow.js
- Layer Configuration: Allow users to edit the number of layers and neurons in each layer of the neural network.
- Hyperparameter Tuning: Enable users to tweak hyperparameters such as learning rate and optimizer settings.
- Live Inference: Run inference and see predictions update live as the user draws on the canvas.

## Style Guidelines:

- Primary color: Midnight blue (#191970) to convey intelligence, and reliability.
- Background color: Dark navy (#2E2E54) - a desaturated and darkened version of the primary, creating a sophisticated backdrop.
- Accent color: Teal (#008080), to highlight active elements.
- Body and headline font: 'Inter', a grotesque-style sans-serif, providing a modern and neutral look suitable for both headlines and body text.
- Code font: 'Source Code Pro' for displaying code snippets.
- Use minimalist, line-based icons for UI elements, with subtle color accents on interaction.
- Divide the application interface into three primary sections: Inference, Prediction, and Model Playground, mirroring the reference screenshot.
- Incorporate subtle transitions and animations to provide feedback on user interactions.
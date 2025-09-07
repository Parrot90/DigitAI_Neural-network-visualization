# Neural Network Visualization

An interactive web application for training and visualizing neural networks on the MNIST handwritten digit recognition dataset.

Please use the Custom Train at First Then it will predict and Please Ignore the AI warning the data overfit 

## 🎯 Features

### **Interactive Neural Network Training**
- Train on real MNIST dataset (60,000 handwritten digits)
- Real-time training progress with accuracy monitoring
- Customizable hyperparameters (learning rate, epochs, batch size)
- AI-powered hyperparameter suggestions

### **Live Model Visualization**
- Interactive neural network architecture display
- Real-time neuron activation during inference
- Dynamic weight visualization with color-coded connections
- Layer-by-layer activity analysis

### **Handwritten Digit Recognition**
- Draw digits on interactive canvas
- Real-time prediction with confidence scores
- Support for all digits (0-9) with high accuracy
- Enhanced preprocessing for better recognition

### **Model Management**
- Save trained models locally
- Load previously trained models
- Export/import functionality for model sharing

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:3000
```

## 🎨 Usage

1. **Train Model**: Configure hyperparameters and train on MNIST data
2. **Draw Digits**: Use the canvas to draw numbers 0-9
3. **Watch Predictions**: See real-time neural network activity
4. **Save Models**: Export trained models for later use

## 🛠 Tech Stack

- **Frontend**: Next.js 15, React 18, TypeScript
- **ML Library**: TensorFlow.js
- **UI Components**: Tailwind CSS, Shadcn/ui
- **AI Integration**: Google Gemini for hyperparameter suggestions

## 📊 Dataset

Uses the official MNIST dataset:
- 60,000 training images
- 10,000 test images  
- 28x28 pixel handwritten digits
- 10 classes (digits 0-9)

## 🔧 Configuration

Set up environment variables in `.env.local`:
```bash
GEMINI_API_KEY=your_api_key_here  # Optional for AI suggestions
```

## 🧠 Model Architecture

Default neural network:
- Input: 784 neurons (28x28 pixels)
- Hidden: 256 → 128 → 64 neurons
- Output: 10 neurons (digit classes)
- Activation: ReLU + Softmax
- Regularization: Dropout + Batch Normalization

## 📈 Expected Performance

- Training Accuracy: 98%+
- Validation Accuracy: 97%+
- Real-time inference on canvas drawings
- Support for various handwriting styles

## 🎓 Educational Value

Perfect for:
- Understanding neural network training
- Visualizing deep learning concepts
- Experimenting with hyperparameters
- Learning TensorFlow.js basics

Future Scope 

### **Live Model Visualization**
- Interactive neural network architecture display
- Basic neuron activation during inference ⚠️
- Layer configuration and editing
- Real-time training status indicators

### **Planned Enhancements** 🚧
- Dynamic weight visualization with color-coded connections
- Layer-by-layer activity analysis

- Gradient flow visualization during training

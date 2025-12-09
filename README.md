# Live Viz MNIST CNN 🧠

A PyTorch Convolutional Neural Network (CNN) built from scratch to recognize handwritten digits (0-9). 

This project features a **real-time training dashboard** that visualizes the model's "brain" as it learns, showing the loss curve, changing weights, and live predictions on the fly.

##  Features

*   **99% Accuracy**: Achieves high accuracy on the MNIST dataset.
*   **Live Dashboard**: A Matplotlib window that updates in real-time during training.
    *   **Loss Curve**: Watch the error rate drop live.
    *   **Weight Visualization**: See the "filters" of the first convolutional layer evolve.
    *   **Live Predictions**: See the AI guess numbers in real-time (Green = Correct, Red = Wrong).
*   **Progress Tracking**: Terminal scanning bar using `tqdm`.
*   **GPU Acceleration**: Automatically detects and uses CUDA (NVIDIA) or MPS (Mac M-Series) if available.

## How to Run

```bash

Clone this repository or download the CNN.py file.
Open your terminal or command prompt.
pip install torch torchvision matplotlib tqdm numpy
Run the program.

```

Note: The first time you run this, it will automatically download the MNIST dataset (approx. 60MB) to a ./data folder.

 Understanding the Dashboard

When the training starts, a popup window will appear containing three sections:

Top (Loss Curve): This line tracks the "Loss" (error). You want this line to go down and to the right. If it flattens out near 0, the model has finished learning.

Middle (Layer 1 Filters): This strip shows the raw weights of the first Neural Network layer. Initially, it looks like grey static noise. As training progresses, you may see patterns emerge as the AI learns to detect edges and curves.

Bottom (Live Predictions): A sample of 8 images from the current training batch.

Text Color Green: The AI predicted correctly.

Text Color Red: The AI predicted incorrectly.

📝 License

Free to use for educational purposes.

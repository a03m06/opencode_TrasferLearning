# opencode_TrasferLearning

# Task 1: Fine-Tuning a Pretrained Model (Transfer Learning)

This project demonstrates transfer learning by fine-tuning a pretrained ResNet-18 model on the CIFAR-10 dataset and deploying it using Streamlit for real-time image classification.

# Objective

Study transfer learning behavior

Fine-tune a pretrained deep learning model on a new dataset

Modify final layers to adapt to a new classification task

Track training performance and analyze results

Deploy the trained model as a web application

# Dataset

CIFAR-10 Dataset

60,000 RGB images (32×32)

10 classes:

Airplane, Automobile, Bird, Cat, Deer

Dog, Frog, Horse, Ship, Truck

# Model Used

ResNet-18 (Pretrained on ImageNet)

ResNet-18 is a deep convolutional neural network that uses residual connections to improve gradient flow and training stability.

# Dataset Preparation

Images resized and normalized

Converted to PyTorch tensors

Standard ImageNet normalization applied

DataLoader used for batching and shuffling

# Final Layer Modification

The original ResNet-18 model outputs 1000 classes (ImageNet).

To adapt it for CIFAR-10:

The final fully connected layer was replaced

New output layer has 10 neurons

model.fc = nn.Linear(model.fc.in_features, 10)

# Fine-Tuning Strategy

Pretrained convolutional layers were frozen

Only the final classification layer was trained

This reduced training time and computational cost

Enabled effective learning on limited resources (CPU)

# Training Details

Optimizer: Adam

Loss Function: CrossEntropyLoss

Learning Rate: 0.001

Epochs: 3

Hardware: CPU

Training accuracy was recorded after each epoch.

# Performance Tracking

Training accuracy increased steadily

Accuracy vs Epoch plot generated

Final training accuracy ≈ 74%

Test accuracy ≈ 75%

This demonstrates effective knowledge transfer from ImageNet to CIFAR-10.

# Failure Cases

Some misclassifications were observed between visually similar classes such as:

Cat vs Dog

Deer vs Horse

Reasons:

Low image resolution (32×32)

Frozen early convolutional layers

Limited fine-tuning epochs

# Deployment

The trained model was deployed using Streamlit Cloud.

Features:

Upload an image (PNG/JPG)

Model predicts CIFAR-10 class

CPU-safe model loading for cloud deployment

# Repository Structure
transfer_learning_task1/
├── app.py                   # Streamlit app
├── training.ipynb           # Training notebook
├── resnet18_cifar10.pth     # Trained model weights
├── requirements.txt         # Dependencies
├── report.txt               # Detailed experiment report
└── README.md                # Project documentation

# Run Locally
pip install -r requirements.txt
streamlit run app.py

# Key Learnings

Transfer learning enables faster convergence

Pretrained models extract strong visual features

Fine-tuning requires fewer epochs and resources

Deployment completes the full ML lifecycle

# Conclusion

This experiment shows that transfer learning significantly improves training efficiency and accuracy, especially when computational resources and training data are limited.

# Author
Arshi Mittal

Arshi Mittal
Induction Task – Transfer Learning & Deep Learning

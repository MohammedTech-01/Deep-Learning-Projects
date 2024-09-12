# MNIST Handwritten Digit Classification using Neural Network

This project demonstrates how to build, train, and evaluate a deep learning model (Neural Network) to classify handwritten digits from the MNIST dataset. MNIST is a large database of handwritten digits commonly used for training various image processing systems and is considered a classic in machine learning and deep learning.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Model Architecture](#model-architecture)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Confusion Matrix](#confusion-matrix)
8. [Building the Predictive System](#building-the-predictive-system)
9. [Results](#results)
10. [Contributions](#contributions)
11. [License](#license)

## Project Overview

The objective of this project is to classify images of handwritten digits (0–9) using a deep learning model. The MNIST dataset contains 60,000 training images and 10,000 test images. Each image is 28x28 pixels, and the task is to predict the correct digit based on these pixel values.

The deep learning model is implemented using TensorFlow and Keras. It consists of multiple layers including fully connected layers (Dense layers) and activation functions like ReLU and Sigmoid.

## Dataset

The MNIST dataset is included in the Keras library and can be loaded directly using:

```python
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

### Dataset Details:

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28x28 pixels
- **Image Type**: Grayscale (1 channel)
- **Number of Classes**: 10 (digits 0–9)

## Project Structure

```plaintext
├── README.md              # Detailed project description and instructions
├── mnist_digit_classifier.py  # Main Python script for model training and prediction
└── mnist_input.png        # Sample image for testing the predictive system
```

## Setup and Installation

### Prerequisites

Ensure that Python 3.6 or above is installed. You will also need the following libraries:

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- OpenCV (for image pre-processing)

## Model Architecture

The Neural Network is built using Keras Sequential API and consists of the following layers:

1. **Flatten Layer**: Converts the 28x28 input image into a 1D array of size 784.
2. **Hidden Layer 1**: Fully connected (`Dense`) layer with 128 neurons and ReLU activation function.
3. **Hidden Layer 2**: Another fully connected (`Dense`) layer with 128 neurons and ReLU activation function.
4. **Output Layer**: A `Dense` layer with 10 neurons and a Sigmoid activation function. This outputs probabilities for each of the 10 classes (digits 0–9).

### Model Compilation

The model is compiled with the following parameters:

- **Optimizer**: Adam (an adaptive learning rate optimization algorithm)
- **Loss Function**: `sparse_categorical_crossentropy`, as it is a multi-class classification problem.
- **Metrics**: Accuracy to monitor performance during training and testing.

```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

## Model Training

The model is trained on the MNIST dataset for 10 epochs. The training process involves feeding the model batches of images and labels, adjusting the weights through backpropagation, and minimizing the loss function to improve accuracy.

```python
model.fit(X_train, Y_train, epochs=10)
```

## Evaluation

After training, the model's performance is evaluated using the test dataset. The `evaluate` method returns the loss and accuracy of the model on unseen test data.

```python
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy}")
```

The model achieved an accuracy of approximately **97.1%** on the test dataset, which indicates strong performance for handwritten digit classification.

## Confusion Matrix

A confusion matrix is plotted to understand the performance of the model across different digit classes. This matrix shows where the model made correct predictions and where it made mistakes.

```python
from tensorflow.math import confusion_matrix
conf_mat = confusion_matrix(Y_test, Y_pred_labels)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()
```

## Building the Predictive System

A predictive system is implemented to classify handwritten digits from a custom image. OpenCV is used to preprocess the image by converting it to grayscale, resizing it to 28x28, and reshaping it to match the input format required by the model.

```python
input_image_path = '/content/mnist_input.png'
input_image = cv2.imread(input_image_path)
grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
input_image_resize = cv2.resize(grayscale, (28, 28))

# Normalize and reshape the image
input_image_resize = input_image_resize / 255.0
image_reshaped = np.reshape(input_image_resize, [1, 28, 28])

# Predict the label
input_prediction = model.predict(image_reshaped)
input_pred_label = np.argmax(input_prediction)

print('The Handwritten Digit is recognised as:', input_pred_label)
```

The predictive system can be used to classify any handwritten digit provided as an image.

## Results

- **Training Accuracy**: 99.37%
- **Test Accuracy**: 97.1%
- The model performs well in predicting handwritten digits, with minimal misclassification errors.

## Contributions

Feel free to fork this repository, raise issues, and submit pull requests to contribute to improving the model or adding new features.

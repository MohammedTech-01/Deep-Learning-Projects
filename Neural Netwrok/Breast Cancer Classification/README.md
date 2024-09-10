# Breast Cancer Classification with Neural Network

This project demonstrates how to build and train a Neural Network (NN) to classify breast cancer tumors as either malignant or benign using TensorFlow and Keras. The dataset used is the Breast Cancer Wisconsin Dataset, which contains features computed from digitized images of fine needle aspirates (FNA) of breast masses.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Model Architecture](#model-architecture)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Making Predictions](#making-predictions)
8. [Results](#results)
9. [Contributions](#contributions)
10. [License](#license)

## Project Overview

Breast cancer is one of the most common cancers among women worldwide. Early diagnosis of breast cancer plays a crucial role in ensuring treatment success. This project utilizes a simple neural network to predict whether a breast tumor is malignant or benign based on a set of features.

The Neural Network is trained using the Breast Cancer Wisconsin dataset and achieves reasonable accuracy in predicting the tumor class. We employ TensorFlow and Keras for model building and training.

## Dataset

We use the [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) available through the `sklearn` library.

- **Features:** The dataset contains 30 numerical features derived from images of breast masses, such as the radius, texture, smoothness, and symmetry.
- **Labels:** 
  - 0: Malignant
  - 1: Benign

## Project Structure

```plaintext
├── README.md            # Detailed project description and instructions
└── breast_cancer_nn.py   # Main Python script with code for model training and evaluation
```

## Setup and Installation

### Prerequisites

Ensure you have Python 3.6 or higher installed on your machine. You also need the following libraries:

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Model Architecture

The Neural Network model is built using the Keras API from TensorFlow. The architecture includes:

1. **Input Layer**: A `Flatten` layer that transforms the input data (30 features) into a 1D vector.
2. **Hidden Layer**: One fully connected `Dense` layer with 20 neurons and ReLU activation function.
3. **Output Layer**: A `Dense` layer with 2 neurons and a sigmoid activation function. This layer provides a binary classification for malignant (0) or benign (1).

### Model Compilation

We use the following parameters to compile the model:

- **Optimizer**: Adam optimizer for adaptive learning.
- **Loss Function**: `sparse_categorical_crossentropy` since it's a binary classification problem with label encoding.
- **Metrics**: Accuracy to track model performance.

## Model Training

The model is trained on the dataset with a 10% validation split. We run the training for 10 epochs, and the training process is visualized by plotting the accuracy and loss curves.

```python
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
```

## Evaluation

After training, the model is evaluated using the test dataset. The test accuracy and loss are reported.

```python
loss, accuracy = model.evaluate(X_test_std, Y_test)
print(f"Test Accuracy: {accuracy}")
```

## Making Predictions

To make predictions on new data, we process the input by standardizing it (as done during training) and then feed it to the trained model.

Here is an example of predicting whether a tumor is malignant or benign based on input data:

```python
input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
prediction_label = [np.argmax(prediction)]

if(prediction_label[0] == 0):
  print('The tumor is Malignant')
else:
  print('The tumor is Benign')
```

## Results

The trained Neural Network achieved a test accuracy of approximately **95%**, which is satisfactory for this type of problem. The accuracy can be improved further by tuning hyperparameters or using more complex architectures.

## Contributions

Feel free to fork this repository, raise issues, and submit pull requests if you find bugs or want to improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


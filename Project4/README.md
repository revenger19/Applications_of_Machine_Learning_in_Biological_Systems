
# Project-4: Neural Network for Boston Housing Price Prediction

## Overview

This project implements a backpropagation neural network from scratch using **NumPy**. The network has three layers: an input layer, one hidden layer, and an output layer. The purpose is to predict housing prices based on the **Boston Housing** dataset. The implementation supports customizable parameters such as the number of hidden layer neurons and the learning rate. For evaluation, the model uses both **5-fold** and **10-fold cross-validation**. The project reports **Mean Squared Error (MSE)** for both training and validation sets.

## Dataset

The model is trained on the **Boston Housing dataset**, which is provided as `housing.csv`. The dataset contains features such as the average number of rooms per dwelling (`RM`), the percentage of the population considered lower status (`LSTAT`), and the pupil-teacher ratio by town (`PTRATIO`). The target variable is the median value of owner-occupied homes (`MEDV`), scaled by 100,000 for better convergence during training.

### Features Used:
- `RM`: Average number of rooms per dwelling
- `LSTAT`: Percentage of lower status population
- `PTRATIO`: Pupil-teacher ratio by town

### Target:
- `MEDV`: Median value of homes (scaled by 100,000)

## Neural Network Architecture

The neural network consists of the following layers:

- **Input Layer**: 3 neurons corresponding to the 3 features (`RM`, `LSTAT`, `PTRATIO`).
- **Hidden Layer**: User-defined number of neurons (varies across experiments).
- **Output Layer**: 1 neuron (for predicting house prices).

The network uses the **sigmoid activation function** in both the hidden and output layers. The weights and biases are randomly initialized and updated using gradient descent with backpropagation.

### Key Functions:

1. **`forward()`**: Performs a forward pass through the network, computing activations.
2. **`backward()`**: Computes the error and updates the weights using backpropagation.
3. **`train()`**: Trains the model over the specified number of epochs.
4. **`predict()`**: Generates predictions using the trained model.

## Model Training

The model is trained for **1000 epochs** on different configurations. The following cases are tested:

- **Case (a)**: 3 neurons in the hidden layer, learning rate = 0.01
- **Case (b)**: 4 neurons in the hidden layer, learning rate = 0.001
- **Case (c)**: 5 neurons in the hidden layer, learning rate = 0.0001

For each case, the model is trained using both **5-fold** and **10-fold cross-validation**. The MSE for training and validation sets is calculated and reported.

## Cross-Validation

The project uses **K-Fold Cross-Validation** to evaluate model performance. Both **5-fold** and **10-fold** cross-validation are performed for each case:

- **5-Fold Cross-Validation**: The data is split into 5 subsets. The model is trained on 4 subsets and validated on the 5th subset. This process is repeated 5 times, with each subset being used as the validation set once.
- **10-Fold Cross-Validation**: Similar to 5-fold, but the data is split into 10 subsets.

## How to Run the Code

### Requirements:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn

### Steps:

1. Place the provided `housing.csv` file in the working directory.
2. Run the `NN.ipynb` file in a Jupyter notebook or as a Python script.
3. The results for all three cases (a, b, c) will be printed, including MSE values for both 5-fold and 10-fold cross-validation.

### Example Commands:

To train the model for case (a) with 5-fold cross-validation:
```python
input_neurons = X_train.shape[1]  # 3 input features (RM, LSTAT, PTRATIO)
output_neurons = 1  # Single output for regression
hidden_neurons = 3  # Case (a)
learning_rate = 0.01

fold_val_losses, avg_train_loss, avg_val_loss = configure_and_train_nn(input_neurons, hidden_neurons, output_neurons, learning_rate, k_folds=5)
```

## Results

The Mean Squared Error (MSE) values for both training and validation sets are displayed for each case. Below are the general results:

- **Case (a)**: Hidden Neurons = 3, Learning Rate = 0.01
  - **5-Fold**: Training MSE: `<value>`, Validation MSE: `<value>`
  - **10-Fold**: Training MSE: `<value>`, Validation MSE: `<value>`

- **Case (b)**: Hidden Neurons = 4, Learning Rate = 0.001
  - **5-Fold**: Training MSE: `<value>`, Validation MSE: `<value>`
  - **10-Fold**: Training MSE: `<value>`, Validation MSE: `<value>`

- **Case (c)**: Hidden Neurons = 5, Learning Rate = 0.0001
  - **5-Fold**: Training MSE: `<value>`, Validation MSE: `<value>`
  - **10-Fold**: Training MSE: `<value>`, Validation MSE: `<value>`

## Conclusion

This project demonstrates how to implement a neural network from scratch using NumPy and apply it to a real-world regression problem. The project also shows the importance of hyperparameter tuning (number of neurons, learning rate) and the benefits of cross-validation to ensure the model generalizes well to unseen data.

## Files in the Submission

- `NN.ipynb`: The main code file containing the neural network implementation and cross-validation experiments.
- `housing.csv`: The Boston Housing dataset used for training.
- `README.md`: This file containing instructions and details on running the code.

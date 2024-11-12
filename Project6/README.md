# AMLBS Project 6 - Cancer Type Prediction Using Machine Learning Models

## Objective

This project focuses on predicting cancer types based on a given dataset by comparing the performance of various machine learning models. Specifically, we evaluate:
- Support Vector Machines (SVM)
- Random Forest (RF)
- Neural Network (NN) regression

Each model's effectiveness in accurately identifying cancer types is analyzed, with insights into their strengths and weaknesses.

## Dataset Description

The dataset used contains 33 features related to breast cancer, including characteristics such as:
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- etc.

The target variable, `diagnosis`, is encoded as:
- **1** for malignant (`M`)
- **0** for benign (`B`)

The data was preprocessed by encoding the target, splitting into training and testing sets, and standardizing features for optimal model performance.

## Model Analysis

### 1. Support Vector Machines (SVM)
SVMs are effective for cancer prediction due to their clear decision boundaries, especially in high-dimensional spaces.

- **Linear Kernel**: Achieved an accuracy of **94.12%**, showing effective linear separation.
- **RBF Kernel**: Performed well with **89.43% accuracy**, allowing for complex decision boundaries when data isn’t linearly separable.

### 2. Neural Network Regression
Neural networks can capture complex, non-linear relationships, making them suitable for nuanced predictions in cancer diagnosis.

- **Initial Model**: Achieved **97.66% accuracy** with limited iterations (300).
- **Hyperparameter Tuning**: Grid search was employed for tuning, with potential improvements anticipated through more advanced tuning techniques.

### 3. Random Forest Regression
Random Forest provided a balanced approach, handling both linear and non-linear relationships effectively.

- **Performance**: Achieved a near-perfect **training and testing accuracy of 99%**, indicating that it has captured nearly all variations in the dataset without overfitting.

## Model Comparison

| Model           | Accuracy (Test Set) | Key Strengths                              | Key Weaknesses                                      |
|-----------------|---------------------|--------------------------------------------|-----------------------------------------------------|
| SVM (Linear)    | 94.12%              | High accuracy, robust for high-dimensional data | Computationally intensive for large datasets       |
| SVM (RBF)       | 89.43%              | Effective for non-linear separable data    | High resource usage for non-linear data             |
| Random Forest   | 99%                 | Interpretability, minimal tuning required  | Slightly lower accuracy compared to NN in some cases |
| Neural Network  | 97.66%              | Captures complex patterns, non-linear data | Requires tuning, potential convergence issues       |

### Recommended Model
The **Random Forest Regressor** is the most suitable model, achieving the highest accuracy (99%) and balanced performance, with minimal risk of overfitting.

## Real-World Implications

Accurate cancer prediction aids in early diagnosis and personalized treatment plans. In real-world settings:
- **Random Forest** is preferred for interpretability and quick decision-making.
- **Neural Networks** may be ideal for more complex datasets that require learning intricate patterns.

## Conclusion

This analysis underscores the importance of model selection and tuning in medical diagnostics, especially for cancer prediction tasks. Machine learning models, particularly **Random Forest** and **Neural Networks**, demonstrate significant potential in aiding healthcare professionals with early and accurate diagnosis.

## Code and Visualizations

Supporting code and visualizations for this project are available in the associated zip file.

## References

1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. [Machine Learning in Cancer Prediction](https://link.springer.com/)
3. [Hyperparameter Tuning in Neural Networks](https://towardsdatascience.com/)

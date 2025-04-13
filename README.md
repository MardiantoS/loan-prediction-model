# Loan Approval Prediction

This repository contains Machine Learning models for predicting loan approval status using various classification algorithms from scikit-learn, TensorFlow, and PyTorch.

## Overview

This project is based on the Kaggle competition: [Loan Approval Prediction](https://kaggle.com/competitions/playground-series-s4e10) by Walter Reade and Ashley Chow. The goal is to predict whether a loan will be approved or denied (binary classification) based on various features related to the applicant's profile and loan requirements.

## Dataset

The dataset contains information about loan applicants, including:

- Personal information (age, income, employment length, home ownership)
- Loan details (amount, interest rate, purpose, grade)
- Credit bureau information (credit history length, past defaults)

The target label is 'loan_status'.

## Project Structure

```
loan-approval-prediction/
├── data/
│   └── train.csv
├── notebooks/
│   ├── sklearn_playground.ipynb
│   └── tf_playground.ipynb
├── models/
│   └── loan_model.pth
├── requirements.txt
└── README.md
```

## Models Implemented

Various classification models have been implemented and compared:

#### Scikit-learn
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Decision Tree
7. Stacking Ensemble
8. Multi-Layer Perceptron (Neural Network)

#### TensorFlow 
1. Custom NN model

#### PyTorch
1. Custom NN model

## Model Performance

Model comparison based on Accuracy and ROC AUC scores from cross-validation:

### Scikit-learn Models

| Model                 | Accuracy | ROC AUC |
|-----------------------|----------|---------|
| Logistic Regression   | 0.9141   | 0.7615  |
| Random Forest         | 0.9513   | 0.8527  |
| Gradient Boosting     | 0.9517   | 0.8566  |
| Support Vector Machine| 0.9439   | 0.8312  |
| K-Nearest Neighbors   | 0.9318   | 0.8129  |
| Decision Tree         | 0.9143   | 0.8351  |
| Multi-Layer Perceptron| 0.9280   | N/A     |
| Stacking Ensemble     | 0.9518   | 0.8565  |

### Cross-Validation Results

| Model                 | Mean ROC AUC | Std    |
|-----------------------|--------------|--------|
| Logistic Regression   | 0.9020       | 0.0021 |
| Random Forest         | 0.9343       | 0.0023 |
| Gradient Boosting     | 0.9398       | 0.0027 |
| Support Vector Machine| 0.8916       | 0.0064 |
| K-Nearest Neighbors   | 0.8773       | 0.0048 |
| Decision Tree         | 0.8325       | 0.0083 |

The Gradient Boosting model achieved the highest performance.

### Deep Learning Models

| Model                 | Accuracy | ROC AUC |
|-----------------------|----------|---------|
| TensorFlow Neural Net | 0.9487   | 0.8454  |
| PyTorch Neural Net    | 0.9503   | N/A     |

## Feature Importance

Random Forest feature importance analysis revealed that the top 5 most important features for predicting loan approval are:

1. Loan percentage of income (0.2358)
2. Loan interest rate (0.1186)
3. Person income (0.1065)
4. Loan grade D (0.0916)
5. Loan amount (0.0727)

## Setup and Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Requirements

Python 3.6+ with the following packages:
- pandas
- numpy
- scikit-learn
- tensorflow
- torch
- tqdm
- jupyter

## Future Work

- Implement more advanced feature engineering techniques: PCA
- Try more complex models such as XGBoost, LightGBM
- Improve model interpretability with SHAP values
- Add visualizations of model performance and data distributions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use or reference this code in your work, please cite it as:
"Mardianto Hadiputro, Loan Approval Prediction Models, GitHub repository, 2025. Available at: https://github.com/[your-username]/loan-approval-prediction"

## Acknowledgments

- Walter Reade and Ashley Chow for creating the Kaggle competition
- The scikit-learn, TensorFlow, and PyTorch team for their machine learning libraries

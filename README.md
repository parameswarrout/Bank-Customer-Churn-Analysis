# Bank Customer Churn Prediction

This project focuses on predicting customer churn for a bank using exploratory data analysis (EDA), machine learning models, and deep learning techniques. The dataset contains customer information, and the goal is to identify patterns and build predictive models to classify customers as likely to churn or not.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Model](#deep-learning-model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

Customer churn is a critical problem for businesses, especially in the banking sector. This project aims to:
1. Analyze the dataset to uncover patterns and insights.
2. Build machine learning models to predict churn.
3. Use deep learning to improve prediction accuracy.

---

## Dataset

The dataset used in this project is `Bank Customer Churn Prediction.csv`, which contains the following features:
- **Numerical Features**: `credit_score`, `age`, `tenure`, `balance`, `estimated_salary`, etc.
- **Categorical Features**: `gender`, `country`, etc.
- **Target Variable**: `churn` (1 for churned customers, 0 for non-churned customers).

---

## Exploratory Data Analysis (EDA)

Key steps in EDA include:
1. **Data Cleaning**: Checking for missing values, duplicates, and outliers.
2. **Descriptive Statistics**: Using `.describe()` and `.info()` to understand the dataset.
3. **Visualization**:
   - Histograms for age and credit score distributions.
   - Bar charts and pie charts for gender and churn distributions.
   - Correlation heatmap to identify relationships between features.

---

## Machine Learning Models

The following machine learning models were implemented:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**
5. **XGBoost**
6. **Support Vector Machine (SVM)**

### Preprocessing
- Numerical features were scaled using `StandardScaler`.
- Categorical features were encoded using `OneHotEncoder`.

### Evaluation Metrics
- **Accuracy**
- **F1 Score**
- **Confusion Matrix**
- **ROC Curve and AUC**

### Hyperparameter Tuning
- Grid search was used to optimize the Random Forest model.

---

## Deep Learning Model

A deep learning model was built using TensorFlow and Keras:
- **Architecture**:
  - Input layer with 64 neurons and ReLU activation.
  - Hidden layer with 32 neurons and ReLU activation.
  - Dropout layers to prevent overfitting.
  - Output layer with a sigmoid activation function.
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### Training
- Early stopping was used to prevent overfitting.
- The model was trained for up to 100 epochs with a batch size of 32.

---

## Results

### Machine Learning Models
- Ensemble methods like Random Forest and Gradient Boosting performed the best, achieving high accuracy and F1 scores.
- SVM reduced false positives but had higher false negatives.

### Deep Learning Model
- Achieved competitive accuracy and F1 scores.
- ROC curve and confusion matrix were used to evaluate performance.

### Key Insights
- Age and churn have a positive correlation.
- Active members are less likely to churn.
- Ensemble methods captured complex patterns better than simpler models.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/bank-customer-churn.git
   cd bank-customer-
   ```
2. Install dependecies:
    ```python
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook:
    ```python
    jupyter notebook EDA.ipynb
    ```
### Acknowledgments
This project was created as part of a portfolio to demonstrate skills in data analysis, machine learning, and deep learning. The dataset was sourced from a bank's customer churn prediction problem.
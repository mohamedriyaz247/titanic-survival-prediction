# Titanic Survival Prediction

A GitHub-ready machine learning project that predicts passenger survival on the Titanic dataset.

## Project Overview

This project demonstrates an end-to-end binary classification workflow using Python and scikit-learn.

The original notebook was expanded into a reproducible project with:

- Data inspection and missing-value analysis
- Data cleaning and feature engineering
- Exploratory data analysis (EDA)
- Leakage-safe preprocessing with scikit-learn pipelines
- Logistic Regression
- Random Forest
- Gradient Boosting
- Accuracy, Precision, Recall, F1-score and ROC-AUC
- Confusion matrix and ROC curves
- Random Forest feature importance

## Dataset

The project uses the classic Titanic passenger dataset with **891 rows and 12 original columns**.

Original columns include:

- `PassengerId`
- `Survived` — target
- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Ticket`
- `Fare`
- `Cabin`
- `Embarked`

The original data contains missing values in `Age`, `Cabin`, and `Embarked`.

## Target

`Survived`

- `0` = did not survive
- `1` = survived

## Features Used

The model uses:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`
- `FamilySize`
- `IsAlone`
- `Title`

### Feature Engineering

Two family-related features are created:

- `FamilySize = SibSp + Parch + 1`
- `IsAlone` indicates whether the passenger travelled alone

A `Title` feature is extracted from passenger names and grouped into common titles (`Mr`, `Miss`, `Mrs`, `Master`) and `Other`.

Identifiers and raw high-cardinality fields such as `PassengerId`, `Ticket`, and raw `Name` are not directly used as model inputs.

## Data Preprocessing

The notebook uses a scikit-learn `ColumnTransformer` and `Pipeline`.

### Numeric features

- Missing values → median imputation
- Scaling → `StandardScaler`

### Categorical features

- Missing values → most-frequent imputation
- Encoding → one-hot encoding

Preprocessing is fitted on the training data, helping prevent information leakage from the test set.

## Machine Learning Models

Three models are compared:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

## Evaluation

The dataset is split into:

- **80% training**
- **20% testing**

A fixed `random_state=42` and stratification are used for reproducibility.

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The notebook also produces:

- Survival distribution
- Survival rate by sex
- Survival rate by passenger class
- Age distribution
- Correlation matrix
- Model metric comparison
- Confusion matrix
- ROC curves
- Random Forest feature importance

## Project Structure

```text
titanic-survival-prediction/
│
├── Titanic_Survival_GitHub.ipynb
├── Titanic-Dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Titanic_Survival_GitHub.ipynb
```

Run the cells from top to bottom.

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn
- jupyter

## Skills Demonstrated

- Python
- Pandas
- NumPy
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing
- Classification
- Logistic Regression
- Random Forest
- Gradient Boosting
- Model Evaluation
- Data Visualization
- scikit-learn Pipelines
- Git/GitHub

## Notes and Limitations

The results are specific to the notebook's fixed train/test split and should not be interpreted as guaranteed performance on unseen datasets.

The Titanic dataset is a historical, relatively small dataset and is mainly useful here for demonstrating a complete machine-learning workflow.

## Author

**Mohamed Riyaz**

B.Tech (Computer Science and Engineering)

GitHub: https://github.com/mohamedriyaz247

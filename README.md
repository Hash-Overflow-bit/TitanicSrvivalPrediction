# Titanic Survival Prediction

A machine learning model to predict who will survive or die on the Titanic disaster.

## Overview

This project implements a comprehensive machine learning solution to predict passenger survival on the Titanic using various features like passenger class, sex, age, fare, and family size. The model uses advanced feature engineering and compares multiple algorithms to achieve optimal prediction accuracy.

## Features

- **Data Preprocessing**: Handles missing values intelligently
- **Feature Engineering**: 
  - Extracts titles from names
  - Creates age groups
  - Calculates family size
  - Engineers fare bins
  - One-hot encodes categorical variables
- **Multiple ML Algorithms**:
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
  - Support Vector Machine (SVM)
- **Model Comparison**: Automatically selects the best performing model
- **Cross-Validation**: Uses 5-fold cross-validation for robust evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hash-Overflow-bit/TitanicSrvivalPrediction.git
cd TitanicSrvivalPrediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the prediction model:
```bash
python titanic_predictor.py
```

The script will:
1. Load the training and test data
2. Perform exploratory data analysis
3. Engineer features
4. Compare multiple ML models
5. Train the best model
6. Generate predictions
7. Save results to `predictions.csv`

## Dataset

The project includes sample Titanic datasets:
- `train.csv`: Training data with survival labels
- `test.csv`: Test data for making predictions

### Features Used:
- **PassengerId**: Unique identifier
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Name**: Passenger name (used to extract title)
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

## Output

The model generates a `predictions.csv` file with:
- PassengerId
- Survived (0 = died, 1 = survived)

## Model Performance

The model achieves competitive accuracy through:
- Intelligent missing value imputation
- Rich feature engineering
- Ensemble methods
- Cross-validation for generalization

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## License

This project is open source and available for educational purposes.
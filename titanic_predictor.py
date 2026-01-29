"""
Titanic Survival Prediction
A machine learning model to predict survival on the Titanic disaster
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class TitanicPredictor:
    """Main class for Titanic survival prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, train_path='train.csv', test_path='test.csv'):
        """Load training and test datasets"""
        print("Loading data...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        return self.train_df, self.test_df
    
    def explore_data(self):
        """Perform basic exploratory data analysis"""
        print("\n=== Data Exploration ===")
        print("\nFirst few rows:")
        print(self.train_df.head())
        
        print("\nData Info:")
        print(self.train_df.info())
        
        print("\nSurvival Rate:")
        survival_rate = self.train_df['Survived'].mean()
        print(f"{survival_rate:.2%}")
        
        print("\nMissing Values:")
        print(self.train_df.isnull().sum())
        
        print("\nSurvival by Gender:")
        print(self.train_df.groupby('Sex')['Survived'].mean())
        
        print("\nSurvival by Class:")
        print(self.train_df.groupby('Pclass')['Survived'].mean())
        
    def engineer_features(self, df, is_train=True):
        """Feature engineering and preprocessing"""
        df = df.copy()
        
        # Extract title from name
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                           'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Fill missing ages based on title
        title_ages = df.groupby('Title')['Age'].median()
        for title in df['Title'].unique():
            if pd.notna(title_ages.get(title)):
                df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = title_ages[title]
        # Fill any remaining missing ages with overall median
        df['Age'].fillna(df['Age'].median(), inplace=True)
        
        # Create age groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Fill missing Embarked with most common
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        # Fill missing Fare with median
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # Create fare bins
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
        
        # Convert categorical to numerical
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # One-hot encode Title
        title_dummies = pd.get_dummies(df['Title'], prefix='Title')
        df = pd.concat([df, title_dummies], axis=1)
        
        # One-hot encode AgeGroup
        age_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
        df = pd.concat([df, age_dummies], axis=1)
        
        # One-hot encode FareBin
        fare_dummies = pd.get_dummies(df['FareBin'], prefix='FareBin')
        df = pd.concat([df, fare_dummies], axis=1)
        
        return df
    
    def prepare_features(self, df, is_train=True):
        """Select and prepare features for modeling"""
        # Select features
        feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                       'Embarked', 'FamilySize', 'IsAlone']
        
        # Add engineered categorical features
        for col in df.columns:
            if col.startswith(('Title_', 'AgeGroup_', 'FareBin_')):
                feature_cols.append(col)
        
        X = df[feature_cols]
        
        if is_train:
            self.feature_columns = feature_cols
            y = df['Survived']
            return X, y
        else:
            # Ensure test data has same columns as train
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            return X[self.feature_columns]
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the prediction model"""
        print(f"\n=== Training {model_type} model ===")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42,
                                               max_depth=5, min_samples_split=10)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Validate
        y_pred = self.model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Died', 'Survived']))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return accuracy
    
    def compare_models(self, X, y):
        """Compare different models"""
        print("\n=== Comparing Models ===")
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        # Use a temporary scaler for comparison only
        temp_scaler = StandardScaler()
        X_scaled = temp_scaler.fit_transform(X)
        
        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5)
            results[name] = scores.mean()
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = max(results, key=results.get)
        print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
        return best_model_name
    
    def predict(self, X_test):
        """Make predictions on test data"""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions
    
    def save_predictions(self, predictions, output_file='predictions.csv'):
        """Save predictions to CSV file"""
        output = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Survived': predictions
        })
        output.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")
        print(f"Predicted {predictions.sum()} survivors out of {len(predictions)} passengers")
        print(f"Survival rate: {predictions.mean():.2%}")


def main():
    """Main execution function"""
    print("=" * 60)
    print("TITANIC SURVIVAL PREDICTION")
    print("=" * 60)
    
    # Initialize predictor
    predictor = TitanicPredictor()
    
    # Load data
    train_df, test_df = predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Engineer features
    print("\n=== Feature Engineering ===")
    train_processed = predictor.engineer_features(train_df, is_train=True)
    test_processed = predictor.engineer_features(test_df, is_train=False)
    
    # Prepare features
    X_train, y_train = predictor.prepare_features(train_processed, is_train=True)
    X_test = predictor.prepare_features(test_processed, is_train=False)
    
    print(f"\nFeatures used: {len(predictor.feature_columns)}")
    print(f"Feature names: {predictor.feature_columns}")
    
    # Compare models
    best_model = predictor.compare_models(X_train, y_train)
    
    # Train best model
    model_map = {
        'Random Forest': 'random_forest',
        'Logistic Regression': 'logistic_regression',
        'Gradient Boosting': 'gradient_boosting',
        'SVM': 'svm'
    }
    predictor.train_model(X_train, y_train, model_type=model_map[best_model])
    
    # Make predictions
    print("\n=== Making Predictions ===")
    predictions = predictor.predict(X_test)
    
    # Save predictions
    predictor.save_predictions(predictions)
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

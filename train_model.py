"""
Script for training and saving logistic regression model
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_save_model():
    """Trains the model and saves it with scaler and encoders"""
    
    # Load data
    train_df = pd.read_csv('train.csv')
    
    # Create copy for processing
    train_data = train_df.copy()
    
    # Handle missing values
    # Fill numeric columns with median
    numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if train_data[col].isnull().any():
            median_val = train_data[col].median()
            train_data[col].fillna(median_val, inplace=True)
    
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if train_data[col].isnull().any():
            mode_val = train_data[col].mode()[0]
            train_data[col].fillna(mode_val, inplace=True)
    
    # Encode categorical variables
    le_dict = {}
    categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    
    for col in categorical_features:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        le_dict[col] = le
    
    # Separate features and target variable
    X = train_data.drop(['id', 'loan_status'], axis=1)
    y = train_data['loan_status']
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    log_reg.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    y_val_pred = log_reg.predict(X_val_scaled)
    y_val_pred_proba = log_reg.predict_proba(X_val_scaled)[:, 1]
    
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print("Validation Results:")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*50)
    
    # Save model
    model_data = {
        'model': log_reg,
        'scaler': scaler,
        'le_dict': le_dict,
        'feature_names': feature_names
    }
    
    with open('credit_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\nModel saved to credit_model.pkl")

if __name__ == '__main__':
    train_and_save_model()

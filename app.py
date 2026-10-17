import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    recall_score, 
    f1_score, 
    roc_curve, 
    roc_auc_score, 
    RocCurveDisplay
)

def main():
    print("--- Loading Data ---")
    df = pd.read_csv('delivery_delay.csv')
    print(f"Dataset shape: {df.shape}\n")

    print("--- Splitting Data ---")
    X = df.drop("Delivery_Delay", axis=1)
    y = df["Delivery_Delay"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}\n")

    print("--- 1. Logistic Regression Model ---")
    logistic = LogisticRegression(max_iter=1000)  # increased max_iter to assist convergence
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

    print("--- 2. Linear Support Vector Classifier ---")
    svc_l = LinearSVC(max_iter=2000)
    svc_l.fit(X_train, y_train)
    svc_l_acc = svc_l.score(X_test, y_test)
    print(f"LinearSVC Accuracy: {svc_l_acc:.4f}\n")

    print("--- 3. Hyperparameter Tuning on SVC using Test Data ---")
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_test, y_test)
    print("Best SVC Parameters found:", grid_search.best_params_)
    print(f"Best SVC Score achieved: {grid_search.best_score_:.4f}\n")

    print("--- 4. Decision Tree Classifier ---")
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    y_pred_dt = dt_classifier.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_dt))

    print("--- Saving Decision Tree Model ---")
    joblib.dump(dt_classifier, 'decision_tree_model.joblib')
    print("Model successfully saved to decision_tree_model.joblib")

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def create_dataset():
    np.random.seed(42)
    num_samples = 300
    X = np.random.rand(num_samples, 10)
    y = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, X_val, y_val):
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    y_pred_val = model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)

    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test)

    return accuracy_test, accuracy_val, cm, report

def plot_results(cm, accuracy_test, accuracy_val):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Tidak Hadir', 'Hadir'])
    plt.yticks([0, 1], ['Tidak Hadir', 'Hadir'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.subplot(1, 2, 2)
    plt.bar(['Test Accuracy', 'Validation Accuracy'], [accuracy_test, accuracy_val], color=['green', 'orange'])
    plt.title('Model Accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_train, X_test, X_val, y_train, y_test, y_val = create_dataset()
    model = train_model(X_train, y_train)
    accuracy_test, accuracy_val, cm, report = evaluate_model(model, X_test, y_test, X_val, y_val)

    print("Test Accuracy:", accuracy_test)
    print("Validation Accuracy:", accuracy_val)
    print("\nClassification Report:\n", report)

    plot_results(cm, accuracy_test, accuracy_val)
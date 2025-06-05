from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model():
    _, _, X_test, y_test = load_and_preprocess('data/sign_mnist_train.csv', 'data/sign_mnist_test.csv')
    model = load_model('sign_language_model.h5')
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == "__main__":
    evaluate_model()

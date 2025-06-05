import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(train['label'].values, num_classes=25)

    X_test = test.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y_test = to_categorical(test['label'].values, num_classes=25)

    return X_train, y_train, X_test, y_test

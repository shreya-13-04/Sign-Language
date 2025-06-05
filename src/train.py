from preprocess import load_and_preprocess
from model import build_model

# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess('data/sign_mnist_train.csv', 'data/sign_mnist_test.csv')

# Build model
model = build_model()

# Train model
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# Save model
model.save("sign_language_model.h5")

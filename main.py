import os
import glob
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warning for demonstration

training_data_dir = r'C:\Users\PC\Desktop\Training'
testing_data_dir = r'C:\Users\PC\Desktop\Testing'
K = 6


class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels
        self.classifier = KNeighborsClassifier(n_neighbors=K)
        self.classifier.fit(self.trainingFeatures, self.trainingLabels)

    def predict(self, features):
        """
        Given a list of features vectors of testing examples
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors
        """
        classifier = KNeighborsClassifier(n_neighbors=K)
        classifier.fit(self.trainingFeatures, self.trainingLabels)
        return classifier.predict(features)


def extract_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None, mono=True)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12), axis=1)
    energy = np.mean(librosa.feature.rms(y=y), axis=1)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y), axis=1)

    # Combine all features into a single vector
    features = np.concatenate((mfccs, energy, zero_crossing_rate))
    return features


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def process_training_data(data_dir):
    features = []
    labels = []

    for filename in glob.glob(os.path.join(data_dir, '*.wav')):
        # Extract features
        extracted_features = extract_features(filename)
        features.append(extracted_features)
        # Get the label from the filename (assuming filename format like "Hebron_01.wav")
        label = os.path.basename(filename).split('_')[0]
        labels.append(label)

    # Print extracted labels
    print(f"Extracted labels: {labels}")

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Print encoded labels
    print(f"Encoded labels: {labels}")

    features = np.array(features)
    features, scaler = preprocess(features)

    return features, labels, scaler, label_encoder


def process_testing_data(audio_file_path, scaler):
    # Extract features
    extracted_features = extract_features(audio_file_path)
    extracted_features = scaler.transform([extracted_features])
    return np.array(extracted_features)


def train_mlp_model(features, results):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    # Increase max_iter and adjust learning_rate_init if needed
    model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', max_iter=1000, learning_rate_init=0.001, solver='lbfgs', verbose=True)
    model.fit(features, results)
    return model


def main():
    # Process training data
    X_train, y_train, scaler, label_encoder = process_training_data(training_data_dir)
    # model_nn = NN(X_train, y_train)

    trained_model = train_mlp_model(X_train, y_train)

    while True:
        # Get user input for test file path
        test_file_path = input("Enter the path to the audio file to test: ")

        # Ensure the path is correctly formatted
        test_file_path = test_file_path.strip('\"')

        if not os.path.isfile(test_file_path):
            print("Invalid file path. Please try again.")
            continue

        # Process the test file
        X_test = process_testing_data(test_file_path, scaler)

        # Make predictions
        # predictions = model_nn.predict(X_test)
        # predicted_label = label_encoder.inverse_transform(predictions)

        predictions = trained_model.predict(X_test)
        predicted_label = label_encoder.inverse_transform(predictions)

        # Print the predicted accent
        print(f"Predicted Accent: {predicted_label[0]}")


if __name__ == "__main__":
    main()

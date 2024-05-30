import os
import glob
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

paths = {}
dir = [
    r'C:\Users\PC\Desktop\Hebron',
    r'C:\Users\PC\Desktop\Nablus',
    r'C:\Users\PC\Desktop\Ramallah_Reef',
    r'C:\Users\PC\Desktop\Jerusalem'
]


def save_paths():
    for i in dir:
        if "Hebron" in i:
            paths['Hebron'] = glob.glob(os.path.join(i, '*.wav'))
        if "Nablus" in i:
            paths['Nablus'] = glob.glob(os.path.join(i, '*.wav'))
        if "Jerusalem" in i:
            paths['Jerusalem'] = glob.glob(os.path.join(i, '*.wav'))
        if "Ramallah_Reef" in i:
            paths['Ramallah_Reef'] = glob.glob(os.path.join(i, '*.wav'))


def read_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr






# Function to extract features from audio files
def extract_features():

    features = {}
    features_list = []
    labels_list = []

    for label, file_paths in paths.items():
        for file_path in file_paths:
            try:
                y, sr = read_file(file_path)

                features['mfccs'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
                features['energy'] = librosa.feature.rms(y=y)
                features['hnr'] = librosa.feature.harmonic_noise_ratio(y=y, sr=sr)

                # Prosodic Features
                features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y=y)
                features['pitch_frequency'] = librosa.pitch.estimate(y=y, sr=sr, fmin=80, fmax=500)

                # Calculate speaking rate (using the number of frames as a proxy for duration)
                features['speaking_rate'] = len(features['mfccs'][0]) / (len(y) / sr)

    return features

# KNN Classifier
def KNN(features, labels, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # K-Nearest Neighbors (KNN) classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("KNN Accuracy:", accuracy_knn)


# Decision Tree Classifier
def decision_tree(features, labels, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    # Decision Tree classifier
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree Accuracy:", accuracy_dt)


# Example usage
save_paths()
features, labels = extract_features()

# Run KNN
KNN(features, labels)

# Run Decision Tree
decision_tree(features, labels)

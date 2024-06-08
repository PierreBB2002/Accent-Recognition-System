import os
import glob
import librosa
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

training_data_dir = r'C:\Users\HP\Desktop\Training'
testing_data_dir = r'C:\Users\HP\Desktop\Testing'
K = 6
root = Tk()
root.title("Spoken project")
root.geometry("850x450+220+120")
root.resizable(False, False)
root.configure(bg="#3776ab")

selected_file_path = StringVar()


class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels
        self.classifier = KNeighborsClassifier(n_neighbors=K)
        self.classifier.fit(self.trainingFeatures, self.trainingLabels)

    def predict(self, features):
        return self.classifier.predict(features)


def extract_features(audio_file_path):
    print(f"Extracting features from {audio_file_path}")
    y, sr = librosa.load(audio_file_path, sr=None, mono=True)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12), axis=1)
    energy = np.mean(librosa.feature.rms(y=y), axis=1)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y), axis=1)

    # Combine all features into a single vector
    features = np.concatenate((mfccs, energy, zero_crossing_rate))
    return features


def evaluate(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    # to print the confusion matrix
    cmatrix = confusion_matrix(labels, predictions)
    return accuracy, precision, recall, f1, cmatrix


def preprocess(features):
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
        label = os.path.basename(filename).split('_')[0]
        labels.append(label)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Print encoded labels
    print(f"Encoded labels: {labels}")

    features = np.array(features)
    features, scaler = preprocess(features)

    return features, labels, scaler, label_encoder


def preprocess_testing_data(data_dir, scaler):
    features = []
    labels = []
    file_paths = []

    for filename in glob.glob(os.path.join(data_dir, '*.wav')):
        extracted_features = extract_features(filename)
        scaled_features = scaler.transform([extracted_features])
        features.append(scaled_features[0])

        label = os.path.basename(filename).split('_')[0]
        labels.append(label)
        file_paths.append(filename)

    return np.array(features), labels, file_paths


def process_testing_data(audio_file_path, scaler):
    # Extract features
    extracted_features = extract_features(audio_file_path)
    extracted_features = scaler.transform([extracted_features])
    return np.array(extracted_features)


def train_mlp_model(features, results):
    model = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', max_iter=1000, learning_rate_init=0.001,
                          solver='adam', verbose=True)
    model.fit(features, results)
    return model


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    selected_file_path.set(file_path)
    print(f"Selected file: {file_path}")
    file_path_label.config(text=file_path)


def analyze():
    try:
        selected_technique = technique_combobox.get()
        if selected_technique == "KNN":
            print("knn")
            model_nn = NN(X_train, y_train)
            model = model_nn.classifier
        elif selected_technique == "MLP":
            print("mlp")
            model = train_mlp_model(X_train, y_train)
        else:
            print("Invalid selection")
            result_label.config(text="Invalid selection")
            return

        test_file_path = selected_file_path.get()

        if not os.path.isfile(test_file_path):
            print("Invalid file path. Please try again.")
            result_label.config(text="Invalid file path. Please try again.")
            return

        X_test = process_testing_data(test_file_path, scaler)

        predictions = model.predict(X_test)
        predicted_label = label_encoder.inverse_transform(predictions)

        print(f"Predicted Accent: {predicted_label[0]}")
        result_label.config(text=f"Accent: {predicted_label[0]}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        result_label.config(text=f"Error: {e}")


def test_all_files():
    try:
        # Get the selected machine learning technique
        selected_technique = technique_combobox.get()
        if selected_technique == "KNN":
            print("knn")
            model_nn = NN(X_train, y_train)
            model = model_nn.classifier
        elif selected_technique == "MLP":
            print("mlp")
            model = train_mlp_model(X_train, y_train)
        else:
            print("Invalid selection")
            result_label.config(text="Invalid selection")
            return

        if not test_file_paths:
            print("No .wav files found in the test directory.")
            result_label.config(text="No .wav files found in the test directory.")
            return

        y_true = []
        y_pred = []

        for i, file in enumerate(test_file_paths):
            # Extract true label from filename
            true_label = y_test_labels[i]
            true_label_encoded = label_encoder.transform([true_label])[0]
            y_true.append(true_label_encoded)

            # Use preprocessed test features
            X_test = np.array([X_test_all[i]])

            # Make predictions
            predictions = model.predict(X_test)
            y_pred.append(predictions[0])

        # Evaluate the model
        accuracy, precision, recall, f1, cmatrix = evaluate(y_true, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{cmatrix}")
        result_label.config(text=f"Accuracy: {round(accuracy, 2)}\nPrecision: {round(precision, 2)}\nRecall: {round(recall, 2)}\nF1 Score: {round(f1, 2)}")
    except Exception as e:
        print(f"Error during testing all files: {e}")
        result_label.config(text=f"Error: {e}")


# Setup GUI for our application
iconImg1 = PhotoImage(file="mainLogo2.png")
root.iconphoto(False, iconImg1)

# header
header = Frame(root, bg="white", width=850, height=95)
header.place(x=0, y=0)

Logo = PhotoImage(file="logo1.png")
Label(header, image=Logo, bg="white").place(x=6, y=7)
Label(header, text="Accent Recognition Application", font="Helvetica 18 bold", bg="white", fg="#333", padx=10, pady=10).place(
    x=90, y=27)

pathLabel = Label(root, text="Selected Path:", font="arial 14 bold", bg="#3776ab", fg="white")
pathLabel.place(x=10, y=160)

file_path_label = Label(root, text="", font="arial 12", bg="white", fg="black", wraplength=380)
file_path_label.place(x=10, y=210)

icon2 = PhotoImage(file="mainLogo2.png")
button1 = Button(root, text="Analyze", compound=LEFT, image=icon2, width=130, font="arial 14", command=analyze)
button1.place(x=30, y=330)

iconSelect = PhotoImage(file="select-file.png")
button2 = Button(root, text="Select File", compound=LEFT, image=iconSelect, width=130, font="arial 14",
                 command=select_file)
button2.place(x=200, y=330)

iconTest = PhotoImage(file="test-file.png")
button3 = Button(root, text="Test All Files", compound=LEFT, image=iconTest, width=150, font="arial 14",
                 command=test_all_files)
button3.place(x=370, y=330)

comboLabel = Label(root, text="Select Technique:", font="arial 14 bold", bg="#3776ab", fg="white")
comboLabel.place(x=600, y=160)

# Combobox for selecting machine learning technique
technique_combobox = ttk.Combobox(root, values=["KNN", "MLP"], font="arial 14", state="readonly", width=10)
technique_combobox.place(x=600, y=210)
technique_combobox.current(0)

result_here = Label(root, text="Result:", font="arial 14", bg="#3776ab", fg="white")
result_here.place(x=600, y=280)

# Label to show result
result_label = Label(root, text="", font="arial 14", bg="white", fg="black", width=18)
result_label.place(x=600, y=330)

print("Analyzing...")

# Process training data
print("Training files:")
X_train, y_train, scaler, label_encoder = process_training_data(training_data_dir)

# Preprocess testing data
print("testing files: ")
X_test_all, y_test_labels, test_file_paths = preprocess_testing_data(testing_data_dir, scaler)

root.mainloop()

import os
import numpy as np
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Function to load images from a folder
def load_images_from_folder(folder):
    X, y = [], []
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                if image_file.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(class_path, image_file)
                    try:
                        image = io.imread(image_path)
                        if image.shape[-1] == 4:  # Check for transparency (4 channels)
                            image = image[:, :, :3]  # Convert to RGB (discard alpha channel)
                        image = transform.resize(image, (64, 64), anti_aliasing=True)
                        X.append(image)
                        y.append(class_name)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    return X, y

# Step 1: Load and preprocess the images
data_dir = "C:\\Users\\hp\\OneDrive\\Bureau\\food_model"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
validation_dir = os.path.join(data_dir, "validation")

X_train, y_train = load_images_from_folder(train_dir)
X_test, y_test = load_images_from_folder(test_dir)
X_val, y_val = load_images_from_folder(validation_dir)

# Check if there are valid images
if not X_train or not X_test or not X_val:
    print("No valid images found in the specified directories.")
    exit()

# Steps 2: Label encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)

# Step 3: Flatten the images into 1D arrays
X_train = np.array(X_train).reshape(len(X_train), -1)
X_test = np.array(X_test).reshape(len(X_test), -1)
X_val = np.array(X_val).reshape(len(X_val), -1)

# Step 4: Split the data into training, testing, and validation sets
# (You may adjust the test_size and random_state as needed)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Step 5: Train a machine learning model (SVM classifier)
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Step 7: Serialize the trained model to a .pkl file using pickle
model_filename = "food_classifier_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
print(f"Model saved as {model_filename}")

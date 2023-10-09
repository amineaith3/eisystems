import os
import numpy as np
from skimage import io, transform
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    try:
        image = io.imread(image_path)
        if image.shape[-1] == 4:  # Check for transparency (4 channels)
            image = image[:, :, :3]  # Convert to RGB (discard alpha channel)
        image = transform.resize(image, (64, 64), anti_aliasing=True)
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Load the trained model from the .pkl file
model_filename = "food_classifier_model.pkl"
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Load and preprocess a sample image for testing
sample_image_path = "C:\\Users\\hp\\OneDrive\\Bureau\\food_model_1\\validation\\capsicum\\capsicum_1.jpg"  # Replace with the path to your image
sample_image = load_and_preprocess_image(sample_image_path)

if sample_image is not None:
    # Make predictions using the loaded model
    predicted_class = model.predict([sample_image.flatten()])
    
    # Load the class names from the subdirectories of the validation directory
    validation_dir = "C:\\Users\\hp\\OneDrive\\Bureau\\food_model_1\\validation"
    class_names = os.listdir(validation_dir)
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class[0]]
    
    # Display the sample image and the predicted class
    plt.imshow(sample_image)
    plt.title(f"Predicted Class: {predicted_class_name}")
    plt.axis('off')
    plt.show()
else:
    print("Image loading and preprocessing failed.")

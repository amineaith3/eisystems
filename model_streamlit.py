import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the pre-trained food classifier model
with open('food_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to classify the food image
def classify_food(image):
    # Preprocess the image (resize, normalize, etc.) as required by your model
    # Here, we assume your model expects an image of a specific size, e.g., (224, 224)
    image = image.resize((224, 224))
    # Convert the image to a numpy array and preprocess further if needed
    image = np.array(image)
    # Make predictions using the model
    prediction = model.predict(image.reshape(1, 224, 224, 3))
    return prediction

# Streamlit app
st.title("Food Type Classifier")

# Upload an image
uploaded_image = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Classify the food type when a button is clicked
    if st.button("Classify Food Type"):
        # Open and process the image
        image = Image.open(uploaded_image)
        prediction = classify_food(image)
        
        # Define your food classes or labels here; replace with your own classes
        food_classes = ["apple", "banana", "beetroot", "bell pepper", "cabbage",
                        "capsicum", "carrot", "cauliflower", "chilli pepper", "corn",
                        "cucumber", "eggplant", "garlic", "ginger", "grapes", "jalepeno",
                        "kiwi", "lemon", "lettuce", "mango", "onion", "orange", "paprika",
                        "pear", "peas", "pineapple", "pomegranate", "potato", "raddish", 
                        "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip",
                        "watermelon"]
        
        # Display the result
        st.write("Predicted Food Type:", food_classes[np.argmax(prediction)])

# Food Image Recognition Project

This project is aimed at recognizing various types of fruits and vegetables using machine learning techniques. It involves creating a model to classify food images and building a user-friendly interface with Streamlit for easy classification.

## Model Creation (model_food.py)

The `model_food.py` script is responsible for creating and training the machine learning model. It takes advantage of a carefully curated dataset to recognize different types of fruits and vegetables. The model is then saved as a `.pkl` file for later use.

## Model Deployment with Streamlit (model_streamlit.py)

The `model_streamlit.py` script leverages the trained model and integrates it with Streamlit, a powerful Python library for creating web applications. Users can upload an image of a fruit or vegetable, and the model will classify it into one of the predefined categories.

### Usage

1. Clone the repository.
2. Install the required dependencies with `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run model_streamlit.py`.
4. Upload a food image to get it classified.

## Testing Model Serialization (test_pkl.py)

The `test_pkl.py` script is used to verify the serialization of the model to a `.pkl` file. It's crucial to ensure that the model can be saved and loaded correctly, making it more accessible for deployment.

## Dataset

We used a publicly available dataset for training our food image recognition model. You can access the dataset by following this link:

[Dataset: Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

Please adjust the dataset's location as needed for your project.

## Results Image

![Food Recognition Result](https://github.com/amineaith3/eisystems/assets/91127128/6dac76c9-75af-468a-82f5-ce4f6f121368)

This image showcases a sample result of the food recognition process.

## Contact

If you have any questions or suggestions, please feel free to contact me at amineaithamma2004@gmail.com.

Happy food recognition!

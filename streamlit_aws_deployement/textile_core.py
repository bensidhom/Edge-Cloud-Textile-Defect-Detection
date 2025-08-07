from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained model
model = load_model('TILDA_model_efficientNet-B0.h5')

# Class labels for textile classification
class_labels = ['Good', 'Hole', 'Objects', 'Oil Spot', 'Thread Error']

def predict_image(file_path):
    """
    Function to preprocess an image and predict its class using the loaded model.

    Args:
        file_path (str): Path to the image file.

    Returns:
        tuple: Predicted class label and confidence score.
    """
    try:
        # Load and preprocess the image
        img = load_img(file_path, target_size=(64, 64))  # Adjust size as needed
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess input

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_confidence = predictions[0][predicted_class]

        # Map the predicted class index to the corresponding label
        return class_labels[predicted_class], predicted_confidence
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

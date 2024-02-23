import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Loading the trained model
model = tf.keras.models.load_model("food_classifier_model.h5")

# Function to preprocess an image before feeding it to the model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values between 0 and 1

# Function to make a prediction using the loaded model
def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    if prediction[0][0] >= 0.5:
        return "Junk Food"
    else:
        return "Healthy Food"

image_path_to_predict = "C:\\Users\\kunal\\Desktop\\My Projects\\Image Classifier using CNN\\test.jpg"
prediction_result = predict_image(image_path_to_predict)
print(f"The predicted class for the given image is: {prediction_result}")

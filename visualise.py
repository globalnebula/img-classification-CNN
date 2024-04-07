import tensorflow as tf
from keras.utils import plot_model

# Loading the trained model
model = tf.keras.models.load_model("food_classifier_model.h5")

# Plot the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

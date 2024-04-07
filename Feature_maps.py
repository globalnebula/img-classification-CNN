import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("food_classifier_model.h5")

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def visualize_feature_maps(model, image_path):
    img = preprocess_image(image_path)
    conv_layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=conv_layer_outputs)
    activations = activation_model.predict(img)

    for i, activation in enumerate(activations):
        num_filters = activation.shape[-1]
        rows = num_filters // 8 + 1
        plt.figure(figsize=(16, rows * 2))
        for j in range(num_filters):
            plt.subplot(rows, 8, j+1)
            plt.imshow(activation[0, :, :, j], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f'Layer {i+1}: {model.layers[i].name} - {num_filters} filters')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

image_path_to_predict = 'test.jpeg'
visualize_feature_maps(model, image_path_to_predict)

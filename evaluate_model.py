import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator


test_path = "C:\\Users\\kunal\\Desktop\\My Projects\\Image Classifier using CNN\\test_images"
img_width, img_height = 150, 150
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_data_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

model = load_model("food_classifier_model1.h5")

y_pred = (model.predict(test_data_generator) > 0.5).astype("int32")


y_true = test_data_generator.classes


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print("Accuracy:", accuracy)
# Calculate the squared error
squared_error = mean_squared_error(y_true, y_pred)

rmse = np.sqrt(squared_error)
print("RMSE:", rmse)
print("Confusion Matrix:")
print(conf_matrix)

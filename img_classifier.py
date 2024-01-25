import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define the paths for your dataset (separate folders for 'junk' and 'healthy' food images)
train_path = "C:\\Users\\kunal\\Desktop\\My Projects\\Image Classifier using CNN\\trainingData"
test_path = "C:\\Users\\kunal\\Desktop\\My Projects\\Image Classifier using CNN\\test_images"

# Image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32
    
# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for test data (no data augmentation)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load and prepare the training data
train_data_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load and prepare the test data
test_data_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data_generator, epochs=10, validation_data=test_data_generator)

# Save the trained model
model.save("food_classifier_model.h5")

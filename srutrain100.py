import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# set the path to the folder containing the images
images_folder = "F:/2303 Sign langauge detection/SIGN LANG SRUSH/train2/"

# set the desired size for the images
desired_size = (224, 224)

# read the images and labels
images = []
labels = []
for letter in range(ord('A'), ord('Z')+1):
    folder = os.path.join(images_folder, chr(letter))
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
            img = cv2.resize(img, desired_size)
            images.append(img)
            labels.append(letter - ord('A'))

images = np.stack(images)
labels = np.array(labels)

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# normalize the pixel values of the images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

# compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.3f}")
print(f"Test Accuracy: {accuracy:.3f}")

# save the model
model.save("sign_language_classifier.h5")
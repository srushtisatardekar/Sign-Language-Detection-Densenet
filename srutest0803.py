import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load the trained model
model = load_model("sign_language_classifier.h5")

# set the desired size for the images
desired_size = (224, 224)

# create a dictionary to map class indices to labels
label_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
              10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
              19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# start capturing video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the video feed
    ret, frame = cap.read()
    
    # convert the frame to color
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # resize the frame to the desired size
    resized = cv2.resize(color, desired_size)
    
    # normalize the pixel values of the image
    normalized = resized.astype('float32') / 255
    
    # reshape the image to match the input shape of the model
    reshaped = np.reshape(normalized, (1, desired_size[0], desired_size[1], 3))
    
    # use the model to make a prediction
    prediction = model.predict(reshaped)
    
    # get the class label for the prediction
    predicted_class = label_dict[int(np.argmax(prediction))]
    
    # display the class label on the frame
    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # display the frame
    cv2.imshow('frame', frame)
    
    # wait for 'q' key to be pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
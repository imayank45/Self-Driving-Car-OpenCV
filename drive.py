# Import the 'socketio' library for implementing WebSocket communication.
import socketio

# Import the 'eventlet' library, which provides tools for building and running concurrent applications.
import eventlet

# Import the 'numpy' library and alias it as 'np' for numerical operations.
import numpy as np

# Import the 'Flask' class from the 'flask' library for building web applications.
from flask import Flask

# Import the 'load_model' function from the 'keras.models' module for loading pre-trained machine learning models.
from keras.models import load_model

# Import the 'base64' library for encoding and decoding base64 data.
import base64

# Import the 'BytesIO' class from the 'io' module for handling binary data as a stream of bytes.
from io import BytesIO

# Import the 'Image' class from the 'PIL' (Python Imaging Library) module for working with images.
from PIL import Image

# Import the 'cv2' module, which is the OpenCV library for computer vision tasks.
import cv2

# Create a Socket.IO server instance
sio = socketio.Server()
 
 
 # Create a Flask web application
app = Flask(__name__) 

# Set a speed limit for the car
speed_limit = 10

# Define a function for preprocessing images
def img_preprocess(img):
    
    # Crop the image to focus on the region of interest
    img = img[60:135,:,:]
    
    # Convert the image to YUV color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Apply Gaussian blur to the image
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    
    # Resize the image to the desired dimensions
    img = cv2.resize(img, (200, 66))
    
    # Normalize pixel values to the range [0, 1]
    img = img/255
    return img
 
 
# Define a function to handle telemetry data received from the client
@sio.on('telemetry')
def telemetry(sid, data):
    
    # Extract speed and image data from the received telemetry data
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    
    # Preprocess the image
    image = img_preprocess(image)
    
    # Expand dimensions to create a batch-like effect for the single image
    image = np.array([image])
    
    # Predict the steering angle using the pre-trained model
    steering_angle = float(model.predict(image))
    
    # Calculate throttle based on speed
    throttle = 1.0 - speed/speed_limit
    
    # Print and send control information
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
 
 
# Define a function to handle the connection event 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Define a function to send control information back to the client
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
# Main block
if __name__ == '__main__':
    
    # Load the pre-trained model
    model = load_model('model/model.h5')
    
    # Create a socket.io middleware for the Flask app
    app = socketio.Middleware(sio, app)
    
    # Start the server to listen for incoming connections on port 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
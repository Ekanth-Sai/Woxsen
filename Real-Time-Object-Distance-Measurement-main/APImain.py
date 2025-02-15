from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

# Your existing code for object detection and distance measurement
# Import necessary libraries, define constants, functions, and the main function

app = Flask(__name__)

# Define your API endpoint
@app.route('/detect-distance', methods=['POST'])
def detect_distance():
    # Get the image data from the request
    data = request.json
    image_data = data.get('image')

    # Decode base64 image data
    decoded_data = base64.b64decode(image_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # Perform object detection and distance measurement
    # Your existing code to detect objects and measure distances

    # Format the result
    result = {
        'detected_objects': detected_objects,  # Your list of detected objects and distances
        'message': 'Object detection and distance measurement successful'
    }

    # Return the result as JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

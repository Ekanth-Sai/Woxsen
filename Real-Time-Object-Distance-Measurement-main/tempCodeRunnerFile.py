import cv2
import os

KNOWN_DISTANCE = 48
PERSON_WIDTH = 15
CUP_WIDTH = 3
MOBILE_WIDTH = 3
SCISSOR_WIDTH = 3

# Initialize YOLO model
yoloNet = cv2.dnn.readNet('E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\yolov4-tiny.weights', 'E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load class names
class_names = []
with open("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\classes.txt", "r") as objects_file:
    class_names = [e_g.strip() for e_g in objects_file.readlines()]

# Function to calculate distance
def cal_distance(f, W, w):
    return (w * f) / W

# Load reference images
person_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\person.jpg")
cup_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\cup.jpg")
mobile_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\mobile.jpg")
scissors_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\scissors.jpg")

# Detect objects in reference images and calculate focal lengths


# Function to detect objects and their distances
def detect_object(image):
    data_dict = {}
    classes, scores, boxes = model.detect(image, 0.4, 0.3)
    for (classid, score, box) in zip(classes, scores, boxes):
        cv2.rectangle(image, box, (0, 0, 255), 2)
        cv2.putText(image, "{}:{}".format(class_names[classid], format(score, '.2f')), (box[0], box[1] - 14), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0), 3)

        # Calculate distance based on object width
        if classid == 0:  # person
            distance = cal_distance(focal_person, PERSON_WIDTH, box[2])
        elif classid == 41:  # cup
            distance = cal_distance(focal_cup, CUP_WIDTH, box[2])
        elif classid == 67:  # cell phone
            distance = cal_distance(focal_mobile, MOBILE_WIDTH, box[2])
        elif classid == 76:  # scissors
            distance = cal_distance(focal_scissor, SCISSOR_WIDTH, box[2])

        # Store object and distance in dictionary
        data_dict[class_names[classid]] = distance

    return image, data_dict

person_data = detect_object(cv2.imread(person_image_path))
focal_person = cal_focalLength(KNOWN_DISTANCE, PERSON_WIDTH, person_data[0][1])

cup_data = detect_object(cv2.imread(cup_image_path))
focal_cup = cal_focalLength(KNOWN_DISTANCE, CUP_WIDTH, cup_data[0][1])

mobile_data = detect_object(cv2.imread(mobile_image_path))
focal_mobile = cal_focalLength(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_data[0][1])

scissor_data = detect_object(cv2.imread(scissors_image_path))
focal_scissor = cal_focalLength(KNOWN_DISTANCE, SCISSOR_WIDTH, scissor_data[0][1])

try:
    capture = cv2.VideoCapture(0)
    while True:
        _, frame = capture.read()

        # Detect objects and distances in real-time video stream
        frame_with_objects, data_dict = detect_object(frame)

        # Display the frame with objects and their distances
        cv2.imshow('frame', frame_with_objects)

        # Print object-distance pairs
        for obj, dist in data_dict.items():
            print("Distance of {} is {} inches".format(obj, dist))

        # Check for exit key press
        exit_key_press = cv2.waitKey(1)
        if exit_key_press == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
except cv2.error:
    print("Error: Select the Webcam or Camera index properly.")

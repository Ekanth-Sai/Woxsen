import cv2
import warnings
import os
import pyttsx3
import termcolor
import time
import math

warnings.filterwarnings("ignore")

KNOWN_DISTANCE = 48
PERSON_WIDTH = 15
CUP_WIDTH = 3
KEYBOARD_WIDTH = 4
MOBILE_WIDTH = 3
SCISSOR_WIDTH = 3
distance = 0

FONTS = cv2.FONT_HERSHEY_TRIPLEX

def detect_object(object):
    classes, scores, boxes = model.detect(object, 0.4, 0.3)
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        cv2.rectangle(object, box, (0, 0, 255), 2)
        cv2.putText(object, "{}:{}".format(str(class_names[classid]), float(score)), (box[0], box[1] - 14), FONTS, 0.6, (0, 255, 0), 3)

        if classid == 0:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 41:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 66:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 67:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 76:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
    return data_list


def cal_distance(f, W, w):
    return (w * f) / W 

def cal_focalLength(d, W, w):
    return (W * d) / w*2


class_names = []
with open("classes.txt", "r") as objects_file:
    class_names = [e_g.strip() for e_g in objects_file.readlines()]

yoloNet = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

person_image_path = os.path.join("src\\person.jpg")
cup_image_path = os.path.join("src\\cup.jpg")
kb_image_path = os.path.join("src\\keyboard.jpg")
moblie_image_path = os.path.join("src\\mobile.jpg")
scissors_image_path = os.path.join("src\\scissors.jpg")

person_data = detect_object(cv2.imread(person_image_path))
person_width_in_rf = person_data[0][1]

mobile_data = detect_object(cv2.imread(moblie_image_path))
mobile_width_in_rf = mobile_data[0][1]

scissor_data = detect_object(cv2.imread(scissors_image_path))
scissor_width_in_rf = scissor_data[0][1]

cup_data = detect_object(cv2.imread(cup_image_path))
cup_width_in_rf = cup_data[1][1]

focal_person = cal_focalLength(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_cup = cal_focalLength(KNOWN_DISTANCE, CUP_WIDTH, cup_width_in_rf)
focal_mobile = cal_focalLength(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_scissor = cal_focalLength(KNOWN_DISTANCE, SCISSOR_WIDTH, scissor_width_in_rf)

detected_objects = []

url  = "http://192.168.0.105:8080/video"

tts_engine = pyttsx3.init()


def speak_detected_objects(detected_objects):
    speech_output = "Detected objects: "
    for obj, dist in detected_objects[:5]:  
        speech_output += f"{obj} at a distance of {dist:.2f} inches. "
    tts_engine.say(speech_output)
    tts_engine.runAndWait()

def determine_position(frame, x1, y1):
    direction_list = []
    box = []
    gesture1 = []
    gesture2 = []
    led1 = []
    led2 = []
    duration = 0
    mid_point_x = 640
    mid_point_y = 360

    """if (1280 - x1 > 0 and mid_point_x - x1 > 0) or (1280 - x1 < 0 and mid_point_x - x1 < 0):  
        if (720 - y1 > 0 and mid_point_y - y1 < 0) or (720 - y1 < 0 and mid_point_y - y1 > 0):  
            return "11"
        else:  # down
            return "10"

    # if (640 < mid_point_x <1200 and 0< mid_point_y <360) or (0 < mid_point_x <640 and 360< mid_point_y <720):
    if (1280 - x1 > 0 and mid_point_x - x1 < 0) or (1280 - x1 < 0 and mid_point_x - x1 > 0):  
        if (720 - y1 > 0 and mid_point_y - y1 < 0) or (720 - y1 < 0 and mid_point_y - y1 > 0):  
           return "01"
        else:  # down
            return "00"""

    if (980 - x1 > 0 and 320 - x1 > 0) or (980 - x1 < 0 and 320 - x1 < 0):  
            if (720 - y1 > 0 and mid_point_y - y1 < 0) or (720 - y1 < 0 and mid_point_y - y1 > 0):  
                return "11"
            else:  
                return "10"

        # if (640 < 320 <1200 and 0< mid_point_y <360) or (0 < 320 <640 and 360< mid_point_y <720):
    if (980 - x1 > 0 and 320 - x1 < 0) or (980 - x1 < 0 and 320 - x1 > 0):  
        if (720 - y1 > 0 and mid_point_y - y1 < 0) or (720 - y1 < 0 and mid_point_y - y1 > 0):  
            return "01"
        else: 
            return "00"


def main():

    KNOWN_DISTANCE = 48
    PERSON_WIDTH = 15
    CUP_WIDTH = 3
    KEYBOARD_WIDTH = 4
    MOBILE_WIDTH = 3
    SCISSOR_WIDTH = 3
    distance = 0

    FONTS = cv2.FONT_HERSHEY_TRIPLEX


    start_time = time.time()
    capture_time = 1
    interval = 7 

    try:
        capture = cv2.VideoCapture(url)
        while True:
            _, frame = capture.read()

            data = detect_object(frame)
            x = 0
            y = 0
            for d in data:
                if d[0] == 'person':
                    distance = cal_distance(focal_person, PERSON_WIDTH, d[1])
                    x, y = d[2]
                elif d[0] == 'cup':
                    distance = cal_distance(focal_cup, CUP_WIDTH, d[1])
                    x, y = d[2]
                elif d[0] == 'cell phone':
                    distance = cal_distance(focal_mobile, MOBILE_WIDTH, d[1])
                    x, y = d[2]
                elif d[0] == 'scissors':
                    distance = cal_distance(focal_scissor, SCISSOR_WIDTH, d[1])
                    x, y = d[2]

                detected_objects.append((d[0], distance))

                cv2.rectangle(frame, (x, y-3), (x+150, y+23), (255, 255, 255), -1)
                cv2.putText(frame, f"Distance:{distance:.2f} inchs", (x+5, y+13), FONTS, 0.45, (255, 0, 0), 2)

                if(determine_position(frame, x, y) == "00"):
                    st = "Take a right"
                elif(determine_position(frame, x, y) == "10"):
                    st = "Take a left"
                elif(determine_position(frame, x, y) == "11"):
                    st = "go ahead and take a left"
                elif(determine_position(frame, x, y) == "01"):
                    st = "go ahead and take a right"
                else:
                    st = "go ahead"

                print("Distance of {} is {:.2f} inchs ".format(d[0], distance ) + st)
                determine_position(frame, x, y)

            cv2.imshow('frame', frame)
            exit_key_press = cv2.waitKey(1)

            if time.time() - start_time >= interval:
                detected_objects.sort(key=lambda x: x[1])
                speak_detected_objects(detected_objects)
                start_time = time.time()
                detected_objects.clear()
                time.sleep(capture_time)

            if exit_key_press == ord('q'):
                break

        capture.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        termcolor.cprint("Select the WebCam or Camera index properly, in my case it is 2", "red")

# Entry point
if __name__ == "__main__":
    main()

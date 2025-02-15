import cv2
import warnings
import os
import termcolor

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


def cal_distance(f,W,w):
    return (w * f) / W 

def cal_focalLength(d, W, w):
    return (W * d) / w*2

class_names = []
with open("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\classes.txt", "r") as objects_file:
    class_names = [e_g.strip() for e_g in objects_file.readlines()]

yoloNet = cv2.dnn.readNet('E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\yolov4-tiny.weights', 'E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\yolov4-tiny.cfg')

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

person_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\person.jpg")
cup_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\cup.jpg")
kb_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\keyboard.jpg")
moblie_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\mobile.jpg")
scissors_image_path = os.path.join("E:\\Ekanth\\ML\\Object_Detection\\Real-Time-Object-Distance-Measurement-main\\src\\scissors.jpg")


person_data = detect_object(cv2.imread(person_image_path))
person_width_in_rf = person_data[0][1]

"""keyboard_data = detect_object(cv2.imread(kb_image_path))
#print(keyboard_data)
keyboard_width_in_rf = keyboard_data[1][1]"""

mobile_data = detect_object(cv2.imread(moblie_image_path))
#print(mobile_data)
mobile_width_in_rf = mobile_data[0][1]

scissor_data = detect_object(cv2.imread(scissors_image_path))
#print(scissor_data)
scissor_width_in_rf = scissor_data[0][1]

cup_data = detect_object(cv2.imread(cup_image_path))
#print(cup_data)
cup_width_in_rf = cup_data[1][1]


focal_person = cal_focalLength(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_cup = cal_focalLength(KNOWN_DISTANCE, CUP_WIDTH, cup_width_in_rf)
#focal_kb = cal_focalLength(KNOWN_DISTANCE, KEYBOARD_WIDTH, keyboard_width_in_rf)
focal_mobile = cal_focalLength(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_scissor = cal_focalLength(KNOWN_DISTANCE, SCISSOR_WIDTH, scissor_width_in_rf)

detected_objects = []

url  = "http://192.168.90.90:8080/video"

try:
    capture = cv2.VideoCapture(url)
    while True:
        _,frame = capture.read()

        data = detect_object(frame) 
        for d in data:
            if d[0] =='person':
                distance = cal_distance(focal_person, PERSON_WIDTH, d[1])
                x,y = d[2]
            elif d[0] =='cup':
                distance = cal_distance(focal_cup, CUP_WIDTH, d[1])
                x, y = d[2]  
            elif d[0] =='cell phone':
                distance = cal_distance(focal_mobile, MOBILE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='scissors':
                distance = cal_distance(focal_scissor, SCISSOR_WIDTH, d[1])
                x, y = d[2]

            detected_objects.append((d[0], distance))

            cv2.rectangle(frame, (x,y-3), (x+150, y+23),(255,255,255),-1)
            cv2.putText(frame,f"Distance:{format(distance,'.2f')}inchs", (x+5,y+13), FONTS, 0.45,(255,0,0), 2)
            
            print("Distance of {} is {} inchs".format(d[0],distance))

        cv2.imshow('frame',frame)
        exit_key_press = cv2.waitKey(1)

        if exit_key_press == ord('q'):
            break

    capture.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error:
    termcolor.cprint("Select the WebCam or Camera index properly, in my case it is 2","red")


for obj, dist in detected_objects:
    print(f"Detected {obj} at a distance of {dist} inches")

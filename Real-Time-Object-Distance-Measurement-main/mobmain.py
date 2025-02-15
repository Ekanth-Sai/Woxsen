from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2
from main2 import detect_object

class ObjectDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)
        return self.layout

    def update_frame(self, dt):
        # Capture frame from camera using OpenCV
        ret, frame = self.capture.read()

        # Perform object detection on the frame
        detected_objects = detect_object(frame)

        # Display detected objects on the frame
        for obj, dist in detected_objects:
            cv2.putText(frame, f"{obj}: {dist:.2f} inches", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert the frame to texture and update the Kivy Image widget
        if ret:
            self.image_widget.texture = self.texture_from_frame(frame)

    def texture_from_frame(self, frame):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def on_start(self):
        # Open camera using OpenCV
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def on_stop(self):
        # Release camera when the app is closed
        self.capture.release()

if __name__ == '__main__':
    ObjectDetectionApp().run()

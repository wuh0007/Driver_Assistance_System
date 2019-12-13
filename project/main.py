"""This is .py file for combining raod lane detection & vehicle detection & warning"""

import numpy as np
import cv2
from PIL import Image
from timeit import default_timer as timer
from vehicle_detection import YOLO
from lane_detection import main_pipline
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# function for continuously detect a video
def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        if frame is None:
            break   
        
        image = Image.fromarray(frame) 
        b, g, r = image.split()
        image = Image.merge("RGB", (r, g, b))
        image3, location = yolo.detect_image(image)
        r,g,b = image3.split()
        image3 = Image.merge("RGB", (b,g,r))
        image3 = np.asarray(image3)
        result2 = main_pipline(image3, location)  
        
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result2, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result2)
        if isOutput:
            out.write(result2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
    
if __name__ == "__main__":
    yolo = YOLO()
    detect_video(yolo,r'C:\Users\wuh00\OneDrive\Desktop\CPS584_FinalProject\test5.mp4')
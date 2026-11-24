
import numpy as np
import cv2
import time as time
from ultralytics import YOLO
import math

model = YOLO("yolov8n-seg.pt") 

# Henry Hudson Pkwy at W. 138 Street
url = 'https://s9.nysdot.skyvdn.com/rtplive/R11_231/playlist.m3u8'
# canal st
url2 = 'https://s7.nysdot.skyvdn.com/rtplive/R11_258/playlist.m3u8'
# Open a sample video available in sample-videos

TARGET_FPS = 15
frame_time = 1.0 / TARGET_FPS


vcap = cv2.VideoCapture(url2, cv2.CAP_FFMPEG)
source_frame_rate = vcap.get(cv2.CAP_PROP_FPS)
print('sfr',source_frame_rate)
if not vcap.isOpened():
   print("File Cannot be Opened")

# image processing functions
def processFrame(frame, H, W, result):
    img = frame
    # INIT MASKS
    obj1_bg0_mask = np.zeros((H,W), np.uint8)
    object_color_layer_mask = np.zeros_like(img)
    # object_original_layer_mask = 

    outline1_bg0_mask = np.zeros((H,W), np.uint8)
    outline_color_layer_mask = np.zeros_like(img)
    outline_tinted_layer_mask = np.zeros_like(img)

    # CREATE BACKGROUND, B&W
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    bw_img3 = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR) 
    bw_img3_2 = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)

    # print(result)

    # reslt loop 
    if result.masks is not None and result.masks.xy:
        for i, poly in enumerate(result.masks.xy):

            # YOLO result produces float coordinates, cv2 takes int
            polygon_points = poly.astype(np.int32)

            # COLOR LAYERS
            # draw white outline later of polygons
            cv2.polylines(outline_color_layer_mask, [polygon_points], True, (255, 255, 255), thickness=10) # BGR
            # create a color layer wintin polygon mask
            red   = math.sin(i + 0) * 127 + 128
            green = math.sin(i + 2) * 127 + 128
            blue  = math.sin(i + 4) * 127 + 128
            cv2.fillPoly(object_color_layer_mask, [polygon_points], (int(blue), int(green), int(red)))   # BGR ints

            

            # BOOLEAN MASK ASSIGNMENT
            # inside detected obj POLYS, set pixels to 1
            cv2.fillPoly(obj1_bg0_mask, [polygon_points], 1)
            # inside detected obj OUTLINES, set pixels to 1
            cv2.polylines(outline1_bg0_mask, [polygon_points], False, 1, 2)

        # convert bg fg mask to boolean arrays
        background0_foreground1_mask_boolean_array = np.where(obj1_bg0_mask == 1, True, False)
        background0_outline1_mask_boolean_array = np.where(outline1_bg0_mask == 1, True, False)

        # COMBINE LAYERS 
        # replace foreground of base image with color layer 
        bw_img3[background0_foreground1_mask_boolean_array] = object_color_layer_mask[background0_foreground1_mask_boolean_array]
        # composite outlines on top???
        bw_img3[background0_outline1_mask_boolean_array] = outline_color_layer_mask[background0_outline1_mask_boolean_array]
        return bw_img3
    else:
        return img

# video loop 
while(True):
    # log start time
    start = time.perf_counter()

    # Capture frame-by-frame
    # ret is 'return' boolean true if captured, false if not
    ret, frame = vcap.read()
    H, W = frame.shape[:2]
    #print cap.isOpened(), ret
    if not ret or frame is not None:
        # ANALYZE IMAGE WITH MODEL
        # model_time = time.time()
        # print('identifying objects...')
        result = model(frame, conf=0.2, verbose=False)[0]
        # print(f"done in {time.time() - model_time:0.3f}s.")
        # NEW IMAGE
        new_frame = processFrame(frame, H, W, result)

        # Display the resulting frame
        cv2.imshow('frame',new_frame)
        # wait only for whatever time is left in this frame's budget
        elapsed = time.perf_counter() - start
        delay_ms = max(1, int((frame_time - elapsed) * 1000))
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break
    else:
        print("Frame is None")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()
print("Video stop")

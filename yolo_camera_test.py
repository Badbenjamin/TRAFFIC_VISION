
import numpy as np
import cv2
import time as time
from ultralytics import YOLO
import math
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

model = YOLO("yolov8n-seg.pt") 

# Henry Hudson Pkwy at W. 138 Street
url = 'https://s9.nysdot.skyvdn.com/rtplive/R11_231/playlist.m3u8'
# Henry Hudson Pkwy at La Salle Street
url2 = 'https://s7.nysdot.skyvdn.com/rtplive/R11_258/playlist.m3u8'

url3 = 'https://s7.nysdot.skyvdn.com/rtplive/R11_259/playlist.m3u8'
# Open a sample video available in sample-videos
# Henry Hudson Pkwy at W. 150 St
url4 = 'https://s9.nysdot.skyvdn.com/rtplive/R11_235/playlist.m3u8'

# Henry Hudson Pkwy at W. 66 Street
url5 = 'https://s9.nysdot.skyvdn.com/rtplive/R11_242/playlist.m3u8'



vcap = cv2.VideoCapture(url3, cv2.CAP_FFMPEG)

source_frame_rate = vcap.get(cv2.CAP_PROP_FPS)
width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Width: {width}")
print(f"Height: {height}")


TARGET_FPS = source_frame_rate
frame_time = 1.0 / TARGET_FPS
# print('sfr',source_frame_rate)
if not vcap.isOpened():
   print("File Cannot be Opened")

# image processing functions
def processFrame(frame, H, W, result):
    img = frame
    # img_copy = img.copy()
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
            # print(poly)
            # YOLO result produces float coordinates, cv2 takes int
            polygon_points = poly.astype(np.int32)
            cls_id = int(result.boxes.cls[i])
            # COLOR LAYERS
            # draw white outline later of polygons
            cv2.polylines(outline_color_layer_mask, [polygon_points], True, (255, 255, 255), thickness=2) # BGR
            # create a color layer wintin polygon mask
            # red   = math.sin(i + 0) * 127 + 128
            # green = math.sin(i + 2) * 127 + 128
            # blue  = math.sin(i + 4) * 127 + 128

            if  cls_id == 0 or cls_id == 1 or cls_id == 2 or cls_id == 3 or cls_id == 4 or cls_id == 5 or cls_id == 7:
                cv2.fillPoly(object_color_layer_mask, [polygon_points], (int(50), int(205), int(50)))   # BGR ints
            else:
                cv2.fillPoly(object_color_layer_mask, [polygon_points], (int(50), int(50), int(205)))   # BGR ints

            
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
        # img[background0_foreground1_mask_boolean_array] = object_color_layer_mask[background0_foreground1_mask_boolean_array]
        # composite outlines on top???
        # img[background0_outline1_mask_boolean_array] = outline_color_layer_mask[background0_outline1_mask_boolean_array]

        just_objects =  np.full_like(img, (0, 0, 0))
        just_objects[background0_foreground1_mask_boolean_array] = img.copy()[background0_foreground1_mask_boolean_array]
        tinted_objects = cv2.addWeighted(just_objects, 0.5, object_color_layer_mask, 0.5, 0)
        img[background0_foreground1_mask_boolean_array] = tinted_objects[background0_foreground1_mask_boolean_array]

        # M = cv2.moments(polygon_points)

        # if M["m00"] != 0:
        #     cx = int(M["m10"] / M["m00"])
        #     cy = int(M["m01"] / M["m00"])

        #     # Write object number at center
        #     cv2.putText(
        #         frame,
        #         str(i),
        #         (cx, cy),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         2
        #     )
        
        return img
    else:
        return img

# SHOULD I SPLIT THIS UP SO IT DOESN'T FIT ON EVERY FRAME?
def quantize(frame):
    n_colors = 32
    img = frame.copy() 
    img_RBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # normalize pixels from 0-1
    img_RBG_np = np.array(img_RBG, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    h, w, c = original_shape = tuple(img_RBG_np.shape)
    assert c == 3
    image_array = np.reshape(img_RBG_np, (h * w, c))

    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # t0 = time()
    labels = kmeans.predict(image_array)

    def recreate_image(codebook, labels, h, w):
        """Recreate the (compressed) image from the code book & labels"""
        return codebook[labels].reshape(h, w, -1)
    # recreate img, turn 0-1 pixels back into 0-255, set as 8-bit, then convert back to BGR
    return cv2.cvtColor((recreate_image(kmeans.cluster_centers_, labels, h, w)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

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
        # SCALE = 2
        # upscaled_frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)    # or INTER_LANCZOS4
        # result = model(up, imgsz=1280, conf=0.2, verbose=False)[0]
        # contrast_frame = cv2.convertScaleAbs(frame, alpha=2, beta=-10)     # alpha > 1 = more contrast
        # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        # out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        filtered_frame = cv2.bilateralFilter(frame, 14, 50, 50)
        # result = model(filtered_frame, classes=[0,1,2,3,4,5,7], conf=0.15, verbose=False)[0]
        # result = model(filtered_frame, classes=[0,1,2,3,4,5,7], conf=0.15, verbose=False)[0]
        result_w_tracking =model.track(
            frame,
            classes=[0,1,2,3,4,5,7],
            conf=.1,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        # print(f"done in {time.time() - model_time:0.3f}s.")
        # NEW IMAGE
        detections_frame = processFrame(filtered_frame, H, W, result_w_tracking)

        quantized_frame = quantize(detections_frame)

        # blurred = cv2.GaussianBlur(quantized_frame, (15, 15), 0)

        # Display the resulting frame
        cv2.imshow('frame',quantized_frame)
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

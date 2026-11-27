
import numpy as np
import cv2
import time as time
from ultralytics import YOLO
import math
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

model = YOLO("yolov8n-seg.pt") 

# Henry Hudson Pkwy at W. 138 Street
# url = 'https://s9.nysdot.skyvdn.com/rtplive/R11_231/playlist.m3u8'
# Henry Hudson Pkwy at La Salle Street
# url2 = 'https://s7.nysdot.skyvdn.com/rtplive/R11_258/playlist.m3u8'

url_3 = 'https://s7.nysdot.skyvdn.com/rtplive/R11_259/playlist.m3u8'
# Open a sample video available in sample-videos
# Henry Hudson Pkwy at W. 150 St
# url4 = 'https://s9.nysdot.skyvdn.com/rtplive/R11_235/playlist.m3u8'

# Henry Hudson Pkwy at W. 66 Street
url5 = 'https://s9.nysdot.skyvdn.com/rtplive/R11_242/playlist.m3u8'
# Henry Hudson Pkwy at W. 177 Street
url_6 = 'https://s7.nysdot.skyvdn.com/rtplive/R11_267/playlist.m3u8'

# harlem river drive
url_7 = 'https://s7.nysdot.skyvdn.com/rtplive/R11_153/playlist.m3u8'
# ts
# ts_url = 'https://nyctmc.org/api/cameras/1927b469-e2dc-4943-a70c-e6e52fd4c48c/image'
# west st
url_8 = 'https://s9.nysdot.skyvdn.com/rtplive/R11_275/playlist.m3u8'

# tz bridge I87 NB NYSTA Gov. Cuomo Bridge MM 13.3
url_9 = 'https://s9.nysdot.skyvdn.com/rtplive/R8_083/playlist.m3u8'

# at north of croton 
url_10 = 'https://s9.nysdot.skyvdn.com/rtplive/R8_098/playlist.m3u8'


vcap = cv2.VideoCapture(url_3, cv2.CAP_FFMPEG)

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
# split into process frame, 
# - takes frame and result, returns an object of objects for each detected object
# - 
def processFrame(frame, H, W, result):
    img = frame
    # img_copy = img.copy()
    # INIT MASKS
    obj1_bg0_mask = np.zeros((H,W), np.uint8)
    object_color_layer_mask = np.zeros_like(img)

    # reslt loop 
    if result.masks is not None and result.masks.xy and result.boxes.id is not None:
        object_ids = result.boxes.id.int().cpu().tolist()
        masks_xy = result.masks.xy
        for obj_id, poly in zip(object_ids, masks_xy):
            # print(poly)
            # YOLO result produces float coordinates, cv2 takes int
            polygon_points = poly.astype(np.int32)
            # cls_id = int(result.boxes.cls[i])

            cv2.fillPoly(object_color_layer_mask, [polygon_points], (int(50), int(205), int(50))) 

            
            # BOOLEAN MASK ASSIGNMENT
            # inside detected obj POLYS, set pixels to 1
            cv2.fillPoly(obj1_bg0_mask, [polygon_points], 1)

            # label objects with ids
            M = cv2.moments(polygon_points)
            if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Draw object ID
                        cv2.putText(
                            frame,
                            f"ID: {obj_id}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
    # OUTSIDE OF MASKING LOOP
    # convert bg fg mask to boolean arrays
    background0_foreground1_mask_boolean_array = np.where(obj1_bg0_mask == 1, True, False)
    

    # COMBINE LAYERS 
    just_objects =  np.full_like(img, (0, 0, 0))
    just_objects[background0_foreground1_mask_boolean_array] = img.copy()[background0_foreground1_mask_boolean_array]
    tinted_objects = cv2.addWeighted(just_objects, 0.5, object_color_layer_mask, 0.5, 0)
    img[background0_foreground1_mask_boolean_array] = tinted_objects[background0_foreground1_mask_boolean_array]

    return img   

        
            
        
        

# SHOULD I SPLIT THIS UP SO IT DOESN'T FIT ON EVERY FRAME?
def quantize(frame):
    n_colors = 16
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
    
    if not ret or frame is not None:

        # bilateral filter blurs image while preserving edges
        filtered_frame = cv2.bilateralFilter(frame, 14, 50, 50)
        
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

        # Display the resulting frame
        cv2.imshow('frame', quantized_frame)
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

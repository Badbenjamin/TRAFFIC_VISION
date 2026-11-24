
import numpy as np
import cv2
import time

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

# video loop 
while(True):
    # log start time
    start = time.perf_counter()

    # Capture frame-by-frame
    ret, frame = vcap.read()
    #print cap.isOpened(), ret
    if not ret or frame is not None:
        # Display the resulting frame
        cv2.imshow('frame',frame)
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

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision
mamboAddr = "e0:14:d0:63:3d:d0"
mambo = Mambo(mamboAddr, use_wifi=True)
print("trying to connect")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    cap = mamboVision.openVideo()

    while True:
        _, frame = cap.read()
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([38, 86, 0])
        upper_blue = np.array([121, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 5000:
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)


        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

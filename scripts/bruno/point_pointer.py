#!/usr/bin/env python
# -*- coding: utf-8 -*-

from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import operator

# define the lower and upper boundaries in the HSV color space
colorLower = (0, 0, 0)
colorUpper = (225, 255, 160)

##################################################################################
# this function evaluates an image using a color mask and returns the center of mass
# of the objects it masks out
# @param: BGR image
# @return: list of CoM and picture frame
##################################################################################
def imagePointCoords(image_frame): 
    frame = image_frame.copy()

    h = frame.shape[0]
    w = frame.shape[1]

    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=w)      # maccheccazooo??
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    points_coord_list = []

    # only proceed if at least one contour was found
    if len(cnts) <= 0: 
        print("no holes in this frame")
        return None

    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and centroid
    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        else:
            continue

        # center of black pixel cluster
        black_pixels = np.nonzero(mask) 
        center = np.mean(black_pixels, axis=1).astype(int)
        cv2.circle(frame, (center[1], center[0]), 5, (0, 0, 255), -1)       # x and y of center are inverterd for whatever reason
        
        points_coord_list.append(center)
    
    return points_coord_list, frame 

##############################################################################################################
##############################################################################################################
##############################################################################################################

# grabbing frames from reference webcam
vs = VideoStream(src=0).start()

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    frame = vs.read()
    if frame is None:
		break 

    points, output_frame = imagePointCoords(frame)

    cv2.imshow("Frame", output_frame) 
    print(points)
    key = cv2.waitKey(1) & 0xFF

    # if key == ord("n"): 
        #        TODO: QUI DENTRO VA FATTA LA MOVIMANTAZIONE LA MOVIMENTAZIONE 

    # if the 'q' key is pressed, stop the loop
    if key == ord('q'):
		break

# camera release
vs.stop()

# close all windows
cv2.destroyAllWindows()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import operator


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())


# define the lower and upper boundaries in the HSV color space
colorLower = (0, 0, 0)
colorUpper = (225, 255, 32)


# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])


# allow the camera or video file to warm up
time.sleep(2.0)
center_list = []

# keep looping
while True:
	# grab the current frame
	frame = vs.read()
 
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
 
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, colorLower, colorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
 
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		for c in cnts:
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# only proceed if the radius meets a minimum size
			if radius > 30 and radius < 40:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
				center_list.append(center)

	center_list.sort(key = operator.itemgetter(0, 1))
	print(center_list)

	#################################################################################################
	#	fitting a line across points to see if the end-effector joint is misaligned wrt the pannel	#
	#################################################################################################
	if center_list:
		fitted_line = cv2.fitLine(np.asarray(center_list), cv2.DIST_L2, 0, 0.01, 0.01)		#line format is: (vx, vy, x, y) where (vx,vy) is the collinear vector 
		print(fitted_line)

		linepoint1 = (fitted_line[2],fitted_line[3])			#(x,y) of one point on the line

		inclination = fitted_line[1]/fitted_line[0]				# m value of the line
		y_zero = fitted_line[3] - inclination*fitted_line[2]	# q value of the line, second point is (0,q)

		linepoint2 = (0, y_zero)

		cv2.line(frame, linepoint1, linepoint2, (0, 255, 255), 2)	#TODO: fix the fact it only drawn half a line (one of the points of the segment is inside the frame)

	#################################################################################################

	del center_list[:]

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	cv2.imshow("Mask", mask)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
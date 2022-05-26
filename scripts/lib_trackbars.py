#!/usr/bin/env python3

import cv2
import numpy as np

# CURRENTLY UNUSED, MOVED FOR MAKING SOME ROOM

# TODO: used to work inside image_filter.ImagePreprocNode class;
# must resolve issues w/ cv2.waitkey when used inside a "callaback loop" istead of a while

def trackbarFilter(self, noisy='n'):
    if noisy == 'n':
        image = self.noisy_img
    elif noisy == 'd':
        image = self.dnoise_img
    else:
        image = self.sim_img
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if self.inversion_untoggle:
        tbar_mask = cv2.inRange(image.copy(), (self.low_H, self.low_S, self.low_V),
                                (self.high_H, self.high_S, self.high_V))
    else:
        zeromax = (0, 50, 50)  # S=70
        zeromin = (180, 255, 255)
        red_mask_min = cv2.inRange(image.copy(), zeromax, (self.low_H, self.high_S, self.high_V))
        red_mask_max = cv2.inRange(image.copy(), (self.high_H, self.low_S, self.low_V), zeromin)
        tbar_mask = cv2.bitwise_or(red_mask_min, red_mask_max)

    self.masked = cv2.bitwise_and(image, image, mask=tbar_mask)
    self.result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(tbar_mask))


def on_low_H_thresh_trackbar(self, val):
    self.low_H = min(self.high_H - 1, val)
    cv2.setTrackbarPos("low_H", "trackbar", self.low_H)


def on_high_H_thresh_trackbar(self, val):
    self.high_H = max(val, self.low_H + 1)
    cv2.setTrackbarPos("high_H", "trackbar", self.high_H)


def on_low_S_thresh_trackbar(self, val):
    self.low_S = min(self.high_S - 1, val)
    cv2.setTrackbarPos("low_S", "trackbar", self.low_S)


def on_high_S_thresh_trackbar(self, val):
    self.high_S = max(val, self.low_S + 1)
    cv2.setTrackbarPos("high_S", "trackbar", self.high_S)


def on_low_V_thresh_trackbar(self, val):
    self.low_V = min(self.high_V - 1, val)
    cv2.setTrackbarPos("low_V", "trackbar", self.low_V)


def on_high_V_thresh_trackbar(self, val):
    self.high_V = max(val, self.low_V + 1)
    cv2.setTrackbarPos("high_V", "trackbar", self.high_V)


def on_inversion_trackbar(self, val):
    self.inversion_untoggle = val


def initTrackbars(self):
    max_value = 255
    max_value_H = 360 // 2
    self.low_H = 0
    self.low_S = 0
    self.low_V = 0
    self.high_H = max_value_H
    self.high_S = max_value
    self.high_V = max_value
    self.inversion_untoggle = 1
    cv2.namedWindow("trackbar")
    # mask will keep values between [max,min] instead of [min,max]; circular channel
    cv2.createTrackbar("inverted --- normal", "trackbar", 0, 1, self.on_inversion_trackbar)
    cv2.setTrackbarPos("inverted --- normal", "trackbar", 1)
    cv2.createTrackbar("low_H", "trackbar", self.low_H, max_value_H, self.on_low_H_thresh_trackbar)
    cv2.createTrackbar("high_H", "trackbar", self.high_H, max_value_H, self.on_high_H_thresh_trackbar)
    cv2.createTrackbar("low_S", "trackbar", self.low_S, max_value, self.on_low_S_thresh_trackbar)
    cv2.createTrackbar("high_S", "trackbar", self.high_S, max_value, self.on_high_S_thresh_trackbar)
    cv2.createTrackbar("low_V", "trackbar", self.low_V, max_value, self.on_low_V_thresh_trackbar)
    cv2.createTrackbar("high_V", "trackbar", self.high_V, max_value, self.on_high_V_thresh_trackbar)

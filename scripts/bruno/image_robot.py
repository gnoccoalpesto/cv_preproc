#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import superros.transformations as transformations
from visionpylib.cameras import Camera
import PyKDL
import cv2

class Pixel2world(object):

    def __init__(self, relative_transform=PyKDL.Frame() ):
        self.relative_transform = relative_transform

    def getPlaneCoefficients(self):
        return transformations.planeCoefficientsFromFrame(self)

    def createRay(self, point_2d, camera):
        point_2d = np.array([
            point_2d[0],
            point_2d[1],
            1.0
        ]).reshape(3, 1)
        ray = np.matmul(camera.camera_matrix_inv, point_2d)
        ray = ray / np.linalg.norm(ray)
        return ray.reshape(3)

    def rayIntersection(self, ray):
        plane_coefficients = self.getPlaneCoefficients()
        t = -(plane_coefficients[3]) / (
            plane_coefficients[0] * ray[0] +
            plane_coefficients[1] * ray[1] +
            plane_coefficients[2] * ray[2]
        )
        x = ray[0] * t
        y = ray[1] * t
        z = ray[2] * t
        inters = np.array([x, y, z])
        return inters.reshape(3)

    def point2DIntersection(self, point_2d, camera):
        ray = self.createRay(point_2d, camera)
        return self.rayIntersection(ray)
 
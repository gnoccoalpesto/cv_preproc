#!/usr/bin/env python3
"""
AUTHOR: FRANCESCO CERRI
TEAM:   ALMA-X ROVER TEAM
        ALMA MATER STUDIORUM, UNIVERSITÀ DI BOLOGNA

simulation and addition of noise
for navigation camera
"""

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time

class ImageNoiseSimulator:
    """
    TODO: accept parameters at simulator launch
    """
    def __init__(self):
        self.cvbridge=CvBridge()

        camera_dict = { 'HazCam': "/camera/image_raw",
                        'NavCam': "zed2/left_raw/image_raw_color"}
        self.camera_topic = camera_dict['NavCam']
        self.noisy_topic = "/camera/image_raw/noisy"

        self.camera_topic_param=\
            rospy.get_param('/image_preproc/input_camera_topic',self.camera_topic)
        self.noisy_topic_param=\
            rospy.get_param('/image_preproc/noisy_camera_topic',self.noisy_topic)

        self.noise_type=rospy.get_param('/image_preproc/noise_type','ug')
        self.noise_intensity=rospy.get_param('/image_preproc/noise_intensity',4)

        print('listened topic: ' + self.camera_topic)
        print('published topic: ' + self.noisy_topic)

        self.sim_img = np.ndarray
        self.in_dtype = None
        self.camera_resolution = tuple
        self.camera_channels=int

        self.noisy_img = np.ndarray
        self.initNoise()

        self.ave_time=0
        self.iter_count=0


        # ROS I/O
        self.cameraListener = rospy.Subscriber(self.camera_topic, Image, self.cameraCallback)
        self.noisyPublisher = rospy.Publisher(self.noisy_topic, Image, queue_size=1)


    def initNoise(self):
        msg = rospy.wait_for_message(self.camera_topic, Image)
        image = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.in_dtype = image.dtype
        self.camera_resolution = image.shape[:2]
        self.camera_channels=image.shape[2]


    #   #   #   #   #   #   #   #
    # MAIN LOOP
    def cameraCallback(self,img_msg):
        try:
            this_time=time.time()
            # SIMULATED IMAGE AQUISITION
            self.sim_img = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')

            # NOISE ADDITION
            # self.noise_type = rospy.get_param('/image_preproc/noise_type')
            # self.noise_intensity = rospy.get_param('/image_preproc/noise_intensity')
            self.noiseSimualtion(
                noise_type=self.noise_type, intensity_coeff=self.noise_intensity)

            # NOISY IMAGE OUTPUT
            # cv2.imshow('noisy image',self.noisy_img)
            self.noisy_img=cv2.cvtColor(self.noisy_img.copy(),cv2.COLOR_RGB2BGR)
            out_msg=self.cvbridge.cv2_to_imgmsg(self.noisy_img)
            self.noisyPublisher.publish(out_msg)
            # self.updateStatistics(this_time)
        except CvBridgeError:
            # TODO: self.bridgerror
            print("cv bridge error")


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    def noiseSimualtion(self, noise_type=None,intensity_coeff=1):
        """
        simulate and adds to the input stream noise
        :param noise_type: 'uniform','u'; 'gaussian','g'; impulsive 'snp','i'
            NOTE: permitted multiple at once: e.g. noise_type='ug' adds both
        :param intensity_coeff: multiplicative coeff for noise intensity
        :return:
        """
        #TODO: simple sum(+) 3x faster but "saturation"

        #TODO: adding directly to outgoing signal doesn't seem to work
        image=self.sim_img.copy()
        noisy_img = np.zeros_like(image, dtype=np.uint8)
        if 'uniform' in noise_type or 'u' in noise_type:
            intensity=intensity_coeff*2.5
            # Min = np.zeros([1, self.camera_channels])
            Min = -intensity*np.ones([1, self.camera_channels])
            Max=intensity*np.ones([1,self.camera_channels])
            cv2.randu(noisy_img, Min, Max)
            image = cv2.add(image, noisy_img)
        if 'gaussian' in noise_type or 'g' in noise_type:
            intensity=intensity_coeff*2.5
            mean=np.zeros([1,self.camera_channels])
            stddev=intensity*np.ones([1,self.camera_channels])
            cv2.randn(noisy_img, mean, stddev)
            image = cv2.add(image, noisy_img)
        if 'snp' in noise_type or 'i' in noise_type:
            Min = np.zeros([1, self.camera_channels])
            Max = 255 * np.ones([1, self.camera_channels])
            cv2.randu(noisy_img, Min, Max)
            noise_thresh = 250
            _, noise_img = cv2.threshold(noisy_img, noise_thresh, Max[0][0], cv2.THRESH_BINARY)
            image = cv2.add(image, noisy_img)
        #TODO
        #     row, col, ch = image.shape
        #     s_vs_p = 0.5
        #     amount = 0.004
        #     out = np.copy(image)
        #     # Salt mode
        #     num_salt = np.ceil(amount * image.size * s_vs_p)
        #     coords = [np.random.randint(0, i - 1, int(num_salt))
        #               for i in image.shape]
        #     out[coords] = 1
        #     # Pepper mode
        #     num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        #     coords = [np.random.randint(0, i - 1, int(num_pepper))
        #               for i in image.shape]
        #     out[coords] = 0
        # if'speckle'in noise_type or's'in noise_type:
        #     row, col, ch = image.shape
        #     gauss = np.random.randn(row, col, ch)
        #     gauss = gauss.reshape(row, col, ch)
        # image = cv2.add(image, noisy_img)
        # if'poisson'in noise_type or'p'in noise_type:
        #     vals = len(np.unique(image))
        #     vals = 2 ** np.ceil(np.log2(vals))
        #     noisy_img = np.random.poisson(image * vals) / float(vals)
        #     image = cv2.add(image, noisy_img)
        self.noisy_img=image


    #   #   #   #   #   #   #   #   #   #   #   #   #
    # PERFORMANCES
    def updateStatistics(self, time_setpoint):
        """
        running average for cycle time
        pause required for misleading results when awiting for user input
        """
        self.ave_time = (self.ave_time * self.iter_count + time.time() - time_setpoint) / (self.iter_count + 1)
        print('avg. cycle [ms]: {}'.format(np.round(self.ave_time * 1000, 6)), end='\r')
        self.iter_count += 1


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
if __name__ == '__main__':
    node_name = 'noise_simulator'
    rospy.init_node(node_name, anonymous=False)
    print("\nNavCam Image Noise Simulator")
    print('node name: ' + node_name)

    simulator = ImageNoiseSimulator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit(0)
    rospy.loginfo("exiting...")








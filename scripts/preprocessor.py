#!/usr/bin/env python3
"""
AUTHOR: FRANCESCO CERRI
TEAM:   ALMA-X ROVER TEAM
        ALMA MATER STUDIORUM, UNIVERSITÀ DI BOLOGNA

image subsampling and denoising
for navigation camera
"""
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from lib_filters import bilateralFilter, denoisingFilter
import time

class ImagePreprocessor:
    """
    TODO: accept parameters at simulator launch
    """
    def __init__(self):
        self.cvbridge=CvBridge()

        camera_dict = { 'HazCam': "/camera/image_raw",
                        'NavCam': "zed2/left_raw/image_raw_color",
                        'nsyHazCam':"/camera/image_raw/noisy",
                        'prprocHazCam':"/camera/image_raw/preproc"}
        self.input_topic = rospy.get_param(
            '/image_preproc/noisy_camera_topic',camera_dict['nsyHazCam'])

        self.preproc_topic=rospy.get_param(
            '/image_preproc/preproc_camera_topic',camera_dict['prprocHazCam'])
        print('listened topic: ' + self.input_topic)
        print('published topic: ' + self.preproc_topic)

        self.noisy_img = np.ndarray
        self.in_dtype = None
        self.camera_resolution = tuple
        self.camera_channels=int
        self.current_resolution= tuple
        self.print_once_resolution=True

        self.preproc_img = np.ndarray
        self.initPreprocessor()

        self.ave_time=0
        self.iter_count=0

        # ROS I/O
        self.noisyListener = rospy.Subscriber(self.input_topic, Image, self.preprocessCallback)
        self.preprocPublisher = rospy.Publisher(self.preproc_topic, Image, queue_size=1)


    def initPreprocessor(self):
        msg = rospy.wait_for_message(self.input_topic, Image)
        image = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.in_dtype = image.dtype
        self.camera_resolution = image.shape[:2]
        self.camera_channels=image.shape[2]


    #   #   #   #   #   #   #   #
    # MAIN LOOP
    def preprocessCallback(self,img_msg):
        try:
            this_time=time.time()
            # NOISY IMAGE AQUISITION
            self.noisy_img = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')

            # DOWNSAMPLING: RESIZING AND BLURRING
            self.downsampleCamera(2,do_blur=True)
            self.noiseRemotion(do_denoise=True)

            # NOISY IMAGE OUTPUT
            out_msg=self.cvbridge.cv2_to_imgmsg(self.preproc_img)
            self.preprocPublisher.publish(out_msg)
            self.updateStatistics(this_time)
        except CvBridgeError:
            # TODO: self.bridgerror
            print("cv bridge error")


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    # IMAGE PREPROCESS
    def downsampleCamera(self,downsize_coeff,do_blur=True):
        """
        resizes (reduces) input camera image stream self.sim_img by 1/downsize_coeff
        :param downsize_coeff>1
        :param do_blur: if True, in addition it blurs the image accordingly,
            creating an higher level in the image pyramid

        self.camera_resolution: original, self.current_resolution: resized
        """
        new_size=self.camera_resolution[1]//downsize_coeff,self.camera_resolution[0]//downsize_coeff
        if do_blur and downsize_coeff!=1:
            self.noisy_img = cv2.pyrDown(self.noisy_img, dstsize=new_size)
        else:
            self.noisy_img=cv2.resize(self.noisy_img,new_size,interpolation=cv2.INTER_AREA)
        self.current_resolution=new_size[1],new_size[0]
        if self.print_once_resolution:
            self.print_once_resolution=False
            print("reduced resolution: {}".format(self.current_resolution))


    def noiseRemotion(self, do_denoise=True):
        """
        :param do_denoise: if False, noisy image is copied into denoised

        AVAILABLE FILTERS:
            - denoising filter: good performance
            - bilateral: preserves visual markers, other objects of interest
            - gaussian: kinda slow, performance could be better but RT application hence k->sigma shall be small
            - median: could be usefull to preserve color, high smoothing, SHARPER HIST (=~ original)
                ALSO it is possible that it must be applied indep. on each channel

        TESTED COMBINATIONS:
            k_size=5 since online use
            0)denoise_image=self.denoisingFilter(self.bilateralFilter(image,k_size=5,sigma_color=35,sigma_space=35))
            1)denoise_image=self.bilateralFilter(self.denoisingFilter(image),k_size=5,sigma_color=35,sigma_space=35)
            2)denoise_image=self.medianFilter(self.denoisingFilter(image))
            3)denoise_image=self.medianFilter(self.denoisingFilter(self.bilateralFilter(image,k_size=5,sigma_color=35,sigma_space=35)))
            4)denoise_image=self.medianFilter(self.denoisingFilter(self.denoisingFilter(image)))
        NOTE: filters in lib_filters.py

        SELECTED: 1; good mix between speed, thin H hist, edges' sharpness           self.dnoise_img: denoised image


        TODO? different color spaces effect on filtering?

        TODO? could channel denoising & merging faster than full denoising?
          does opencv manage this autonomously (like w/ big kernels)
        """
        if do_denoise:
            image = self.noisy_img.copy()
            denoise_image = bilateralFilter(denoisingFilter(image), k_size=5, sigma_color=45, sigma_space=45)
            self.preproc_img = denoise_image.astype(self.in_dtype)
        else:
            self.preproc_img = self.noisy_img.copy().astype(self.in_dtype)
        # TODO? faster conversion? np.TYPE(IMG), IMG.astype('TYPE), .ASTYPE(np.TYPE)

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
    node_name = 'image_preprocessor'
    rospy.init_node(node_name, anonymous=False)
    print("\nNavCam Image Preprocessor")
    print('node name: ' + node_name)

    preprocessor=ImagePreprocessor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit(0)
    rospy.loginfo("exiting...")








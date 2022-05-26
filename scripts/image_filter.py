#!/usr/bin/env python3
"""
AUTHOR:     FRANCESCO CERRI
            francesco.cerri2@studio.unibo.it

            ALMA-X TEAM
            UNIVERSITÀ DEGLI STUDI DI BOLOGNA ALMA MATER STUDIORUM

THIS CODE EMOVES THE TERRAIN (AND OTHER SIMILAR FEATURES, POSSIBLY)
FROM THE SIMULATION OF "MARSYARD" ARENA (MORE ABOUT THIS LATER)
TO LIGHTEN THE CAMERA STREAM FOR THE FOLLOWING NODES DOWNSTREAM

RELATED WORKS (kinda messy, requires iphyton capable sys):
    https://github.com/alma-x/CV-Alma-X/
--------
INFO ABOUT ROS:
creates a network programs ("nodes"), exchanging messages using peculiar
named communication channels ("topics")
Nodes can subscribe (listen asynchronously) to topics and publish; all this process
is managed by callbacks and threads

HOW ROS WORKS HERE:
(in "main", at file's footer)
once class is instanciated, Subscibers are registered (linked to specific communication channel)

ropsy.spin() starts the LOOP, starting effectively the listening proccess
main loop is fired each time a message on the image topic is received
---------
INFO ABOUT ERC COMPETITION:
"MARSYARD": THE CHALLENGE GROUND FOR EUROPEAN ROVER CHALLENGE
- SIMULATION USES 3D SCANS OF REAL TERRAIN
- TEXTURE OBTAINED USING AERIAL PHOTOGRAPIES OF REAL TERRAIN
- ARTIFICIAL/SIMULATED ILLUMINATION
   TODO: script sun motion in gazebo
----------
INFO ABOUT DRIVING ALGORITHM:
FOR SAKE OF COMPLETENESS, REMOVING THE TERRAIN DOES NOT POSE A RISK SINCE
A DEPTH CAMERA IS ALSO PRESENT.
REMOVING THE TERRAIN HELPS TO (POSSIBLY) IDENTIFY OTHER TERRAIN FEATURES OF A
SMALLER SCALE WRT EFFECTIVE DEPTH CAMERA RESOLUTION USING COLOR AS LONG AS OBJECTS,
MARKERS AND OTHER POINTS OF INTEREST
--------
WHAT THIS CLASS DOES:
0) establish connection with ROS master node
0.1) waits for incoming camera messages
- 2 available in dictionary representing the 2 avail. cameras (rover, zed2)
1) adds noise to simulated view (gaussian+uniform)
2) denoises signal
3) (possible) image analysis to determine some statistics of the histogram(s)
- NOTE: since terrain has a peculiar color, Hue channel is deeply utilized
4) thresholding
- performances of different methods are tested considering correctness, speed,...
- test of an ADAPTING SAMPLING FILTER
4.1) refinement of the mask
- using morphological operators; test of other stuff
5) publish the result for downstream use
TODO: implement the library of features from Alma-X's science team and their
    fancy learning algorithms
####
NOTE: as for now, everything is kinda hardcoded in the main loop, yet methods and functions
 work kinda parametrically for most use cases
-----------------------
IDEAS
    almost constant reddish color (const lighting)->
        clear bimodal histogram on Hue channel

    inherent structure of image->
        mostly sky above and rettain below 2/3 img heigh (duh)

-----------------------
TESTED ON:
intel 12700H + nvidia 3060 mobile
ubuntu 22.04
nvidia drivers 510.60.02

docker 20.10.16

python          3.8.10      REQUIRED
opencv-python   4.5.5.64    REQUIRED
numpy           1.22.3      REQUIRED
ros             noetic      REQUIRED*
rospy           1.15.14     REQUIRED*
cv_bridge       1.16.0      REQUIRED*
lib_filters                 included
lib_histograms              included

SIMULATION (provides all the ingoing data)
https://github.com/EuropeanRoverChallenge/ERC-Remote-Navigation-Sim

NOTE: packages denoted by REQUIRED* are intended for ROS usage
        if not present, use the noROS_NAME.py version (loads a saved video)
"""
# python ROS implementation
import rospy
# ROS message types: required for I/O
from sensor_msgs.msg import Image#,Imu
from std_msgs.msg import String,Bool
# ROS image <--> cv/np image conversion
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
# encapsulates filter definition, histograms analysis tools
from lib_filters import bilateralFilter, denoisingFilter
from lib_histograms import computeHistogram, drawHistogram
#TODO: fix import
#TODO? ALSO import class methods?
import time


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
class ImagePreprocNode:
    """    """
    def __init__(self):
        camera_dict = { 'NavCam': "/camera/image_raw",
                        'HazCam': "zed2/left_raw/image_raw_color"}
        self.camera_topic = camera_dict['HazCam']
        self.filtered_topic = "/camera/image_raw/filtered"
        print('listened topic: ' + self.camera_topic)
        print('published topic: ' + self.filtered_topic)

        self.cvbridge = CvBridge()

        #TODO: "param"_dict={'morph_param':'/image_preproc/morph_param',
        #                     'morph_topic':...}
        self.morph_ops_source='/image_preproc/morph_ops'

        self.add_sample_source='image_preproc/add_request'
        imu_dict = {'imu': "/imu/data_raw",
                    'camera': "/zed2/imu/data"}
        self.imu_topic=imu_dict['imu']
        self.imu_msg = []

        self.sim_img = np.ndarray
        self.in_dtype = None
        self.camera_resolution = tuple
        self.camera_channels=int
        self.current_resolution= tuple
        self.print_once_resolution=True

        self.noisy_img = np.ndarray
        self.dnoise_img = np.ndarray

        self.sky_to_groud=float
        self.sky_image=np.ndarray
        self.ground_image=np.ndarray

        self.MORPH_OPS='c' #odh hdc hdo

        self.res_ada= np.ndarray
        self.res_sample= np.ndarray
        self.res_otsu= np.ndarray
        self.res_range= np.ndarray
        self.res_bal= np.ndarray
        self.ada_mask= np.ndarray
        self.sample_mask= np.ndarray
        self.otsu_mask= np.ndarray
        self.range_mask= np.ndarray
        self.bal_mask= np.ndarray
        self.select_new_sample=False
        self.samples=[]
        self.sample_mask=None
        #   #   #   #   #   #   #   #   #
        self.initFilter()
        # start I/O operations; Subscribe: each time a message comes, Publish: upon request
        self.filteredPublisher = rospy.Publisher(self.filtered_topic, Image, queue_size=1)
        self.cameraListener = rospy.Subscriber(self.camera_topic, Image, self.preprocCallback)
        #TODO? could a timer solve the callback freeze when selecting sample issue?
        # TODO ALSO could to set a greater frequency to decrease overhead
        # self.preprocTimer=rospy.Timer(rospy.Duration(0.01),self.preprocCallback)

        self.morphOpsListener=rospy.Subscriber(self.morph_ops_source,String,self.morphOpsCallback)

        self.addRequestListener=rospy.Subscriber(self.add_sample_source,Bool,self.addSampleCallback)
        # TODO: for remotion requests; could modify the one above
        # self.removeRequestListener=...

        # self.imuListener=rospy.Subscriber(self.imu_topic,Imu,self.imuCallback)

        #   PERFORMANCES   #   #   #   #   #   #
        self.pause_stats=False
        self.iter_stamp=0
        self.iter_counter=0
        self.ave_time=0

        #   KCLUSTERS     #   #   #   #   #   #
        self.n_colors = 12
        self.cl_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.cl_flags = cv2.KMEANS_RANDOM_CENTERS
        #   #   #   #   #   ##   #   #   #   #   ##   #   #   #   #   ##   #   #   #   #   #


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   INITIALIZATION, I/O CALLBACKS, STATISTICS   #   #   #   #   #   #
    def initFilter(self):
        print('initializing...')
        msg = rospy.wait_for_message(self.camera_topic, Image)
        image = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # original dtype is uint8, display also accepts float32 (NOTE: result uint8==float32/255)
        self.in_dtype = image.dtype
        self.camera_resolution = image.shape[:2]
        print("input resolution: %s" % str(self.camera_resolution))
        self.camera_channels=image.shape[2]
        self.current_resolution = self.camera_resolution


    def morphOpsCallback(self,str_msg):
        #TODO version w/ timered/every-cycle rosparam reading (get) overhead?
        """
        whenerever a morph_ops string is received, updates self.MORPH_OPS
        """
        self.MORPH_OPS=str_msg.data


    def addSampleCallback(self,add_sample_msg):
        """
        upon msg received, selects a new Image sample to be removed from filter
         and update the mask
        """
        #TODO: putting cv2.SelectROI() in a callback freezes that window, hence everything
        # since simple threading not working, SPAWNING A CLONE OF PREPROC_LISTENER CB
        # while this one processes input could be an idea
        # remember to kill that new listener once added the sample
        if add_sample_msg:
            self.select_new_sample=True


    #--UNUSED--
    def imuCallback(self,raw_imu):
    # TODO: refine sky recognition using cross imu data:
    #  roll angle sets inclination of division line (horizon tilting)
    #  pitch angle sets distance from the upper corner:
    #   >0 toward x center (width wise direction) (horizon up)
    #   <0 toward y center (height wise direction) == far from x center (horizon down)
    # https://wiki.ros.org/imu_filter_madgwick
        self.imu_msg=raw_imu


    def updateStatistics(self,time_setpoint):
        """
        running average for cycle time
        pause required for misleading results when awiting for user input
        """
        self.iter_stamp+=1
        if not self.pause_stats:
            self.ave_time = (self.ave_time * self.iter_counter + time.time() - time_setpoint) / (self.iter_counter + 1)
            print('avg. cycle [ms]: {}'.format(np.round(self.ave_time * 1000, 6)), end='\r')
            self.iter_counter += 1
        else:
            self.pause_stats=False

#    ##################################################################################
###    ##################################################################################
######    ##################################################################################
    # MAIN LOOP : every time a msg on self.camera_topic received
    def preprocCallback(self, raw_img):
        try:
            # SIMULATED INPUT -- 1 ms ------------------------------------------------------------------
            self.sim_img = self.cvbridge.imgmsg_to_cv2(raw_img, desired_encoding='passthrough')
            self.downsampleCamera(2,do_blur=True)

            # NOISE ADDITION (GAUSSIAN+UNIFORM) -- + 10 ms ---------------------------
            self.noiseSimualtion(noise_type='ug',intensity_coeff=1)

            # DENOISING -- + 4 ms -----------------------------------------------------------------------
            self.noiseRemotion(do_denoise=True)

            # IMAGE ANALYSIS -- + 5 ms ------------------------------------------------------------------
            # self.cameraAnalysis(show_hist=True)

            # self.splitCamera(self.sky_to_groud,hud=False)

            # self.cameraClustering()

            # BACKGROUND FILTERING -- +10 ms ------------------------------------------------------------------------
            this_time = time.time()
            morph_ops=self.MORPH_OPS
            show_result=True
            win_prefix=''
            # self.otsuFilter(show_result=show_result, morph_ops = morph_ops, winname_prefix = win_prefix)
            # self.balancedFilter(show_result=show_result, morph_ops = morph_ops, winname_prefix = win_prefix)
            # self.rangeFilter(show_result=show_result, morph_ops = morph_ops, winname_prefix = win_prefix)
            self.sampleFilter(show_result=show_result, morph_ops = morph_ops, winname_prefix = win_prefix)

            # OUTPUT ------------------------------------------------------------------------------------
            #TODO: superimpose time matermark on output to understand when output stopped

            # SIMULATED
            # self.showChannels(noisy='s',channels='f')
            # self.showHistograms(noisy='s')
            # NOISY
            # self.showChannels(channels='f',noisy='n',winname_prefix='U+G')
            # self.showHistograms(noisy='n')
            # DENOISED
            # self.showHistograms(noisy='d',channels='h')
            # self.showChannels(channels='f')
            # self.showChannels(channels='hsv',h_shift=30)

            out_msg = self.cvbridge.cv2_to_imgmsg(self.res_sample)
            self.filteredPublisher.publish(out_msg)

            ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
            k = cv2.waitKey(1) & 0xFF
            if k == 27: exit(0)

            self.updateStatistics(this_time)
        except CvBridgeError:
            # TODO: self.bridgerror
            print("cv bridge error")

######    ##################################################################################
###    ##################################################################################
#    ##################################################################################

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    # INPUT TOOLS   #   #   #   #   #   #
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
            self.sim_img = cv2.pyrDown(self.sim_img, dstsize=new_size)
        else:
            self.sim_img=cv2.resize(self.sim_img,new_size,interpolation=cv2.INTER_AREA)
        self.current_resolution=new_size[1],new_size[0]
        if self.print_once_resolution:
            self.print_once_resolution=False
            print("reduced resolution: {}".format(self.current_resolution))


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
        channels = image.shape[2]
        noisy_img = np.zeros_like(image, dtype=np.uint8)
        if 'uniform' in noise_type or 'u' in noise_type:
            Min = np.zeros([1, channels])
            intensity=intensity_coeff*2.5
            Max=intensity*np.ones([1,channels])
            cv2.randu(noisy_img, Min, Max)
            image = cv2.add(image, noisy_img)
        if 'gaussian' in noise_type or 'g' in noise_type:
            intensity=intensity_coeff*2.5
            mean=intensity*np.ones([1,channels])
            stddev=intensity*np.ones([1,channels])
            cv2.randn(noisy_img, mean, stddev)
            image = cv2.add(image, noisy_img)
        if 'snp' in noise_type or 'i' in noise_type:
            Min = np.zeros([1, channels])
            Max = 255 * np.ones([1, channels])
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


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   IMAGE PREPROCESSING #   #   #   #   #   #
    def noiseRemotion(self,do_denoise=True):
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
            denoise_image=bilateralFilter(denoisingFilter(image),k_size=5,sigma_color=45,sigma_space=45)
            self.dnoise_img = denoise_image.astype(self.in_dtype)
        else:
            self.dnoise_img = self.noisy_img.copy().astype(self.in_dtype)
        #TODO? faster conversion? np.TYPE(IMG), IMG.astype('TYPE), .ASTYPE(np.TYPE)


    def cameraClustering(self,cl_method='sy',show_result=False):
        """
        clustering routine using k-means
        :param cl_method: used method for clustering
        :param show_result: shows result in a window
        Tested:
            'cv2' or 'cv' or 'c'
            'scipy' or 'sy' or 'y'      (compatible speed, ~80ms overhead)
            'sklearn' or 'sk' or 'k'    (awkwardly slow)

        TODO initialize this offline to find dominant colors of image
        """
        image = self.dnoise_img
        current_resolution=self.current_resolution
        image = image.reshape((-1, 3)).astype('float32')
        if'c'in cl_method:
            _, labels, centers = cv2.kmeans(image, self.n_colors, None, self.cl_criteria, 10, self.cl_flags)
        elif'y'in cl_method:
            centers,labels = kmeans2(image, self.n_colors,10,1,'points','warn',False)
        centers=np.uint8(centers)
        clustered=centers[labels.flatten()]
        h,w=current_resolution
        clustered= clustered.reshape((h,w,self.camera_channels))
        #TODO: dominant colors
        # _, counts = np.unique(labels, return_counts=True)
        # dominant = centers[np.argmax(counts)]
        # label_counts = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        # histogram, _ = np.histogram(cluster.labels_, bins=label_counts)
        # histogrdekam = histogram.astype('float32')
        # histogram /= histogram.sum()
        if show_result:
            cv2.imshow('clustered', cv2.cvtColor(clustered, cv2.COLOR_RGB2BGR))


    #-- UNUSED --
    # def dynamicSplitLevel(self):
    #     """
    #     dynamically obtaining a threshold level to devide sky and ground
    #     :param histogram: input to analize
    #     """
    #     #TODO? which better? otsu, adaptive,malanobis distance, sample var over many imgs,...
    #     split_level=0
    #     return split_level


    def cameraAnalysis(self,show_hist=False,safety_ratio=.2):
        """
        :param show_hist: shows devided histogram of h chan
        :param safety_ratio: enlarges ground window of safety_ratio*ground_window_height
                meant to prevent omission of objects seen from below, into che sky
        obtain details and informations about the histogram of the camera stream

        H channel presents 2 clearly separated peaks: ground vs sky color
            this could be used to separate them and process ground separately,
            saiving hopefully computational time
        basic process: pxs count and ratio

        S chan: - @glacier presents a thick peak
                - @whiteish sand inside crater presents a narrower, higher peak
         -> can use maxima but also pxs counts w/ adaptive/otsu threshold

        TODO tresholding: hardcoded, BUT self.dynamicSplitLevel could be implemented
         dynamic threshold: around each peak center an interval with extension
         depending on pixel ratio in each mode
         BUT: could be better to have a narrow interval for red color
         ALSO: is this really necessary since the clear separation FOR ALL CONDITIONS?

        TODO ISSUE: "false" positives when glacier (white slide inside craters) in frame;
         histogram gets many, almost equally spaced peaks in its "sky side"4
         YET low pass filter could harm also sky, rock peaks

        TODO: as for now, roll amount is not dynamic

        TODO: use this recursively to find out dominant colors using cv2.minMaxLoc()
         local maxima could be usefull for rocks, glacier, mud volcanoes...
        """
        roll_amount=30
        image = self.dnoise_img
        hsv_image=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2HSV)
        # range_normalization helps with (de)noisy pxs count
        _,_,histogram= computeHistogram(hsv_image,'h',range_normalize=True)
        # brings the 2 sides of red peak (around 180==0) altogether
        histogram = np.roll(histogram, roll_amount)
        # INTENSITY RANGES for s_l=80: ground [151-180=0-50], sky[51-150]
        split_level=80
        hist_ground,hist_sky=histogram[:split_level,...],histogram[split_level:,...]
        count_ground=np.sum(hist_ground)
        count_sky=np.sum(hist_sky)
        self.sky_to_groud=(1-safety_ratio)*count_sky/(count_sky+count_ground)
        # global maxima
        # gmax_ground=256# NORMALIZED RANGE [0, 256]
        # gmax_sky=np.max(hist_sky)
        # arg_gmax_ground=(np.argmax(hist_ground)+151)%180
        # arg_gmax_sky=(np.argmax(hist_sky)+51)%180 if count_sky!=0 else 'UNDEF'
        # also: np.where(ARR=MAX(ARR))
        if show_hist:
            hist_img=drawHistogram(histogram,img_zoom=3,shift_percent=1)
            hist_img=hist_img[hist_img.shape[0]//2:,:]

            noisy_img=self.noisy_img.copy()
            hsv_noisy=cv2.cvtColor(noisy_img,cv2.COLOR_RGB2HSV)
            _,_,noisy_hist= computeHistogram(hsv_noisy,'h',range_normalize=True)
            noisy_hist = np.roll(noisy_hist, roll_amount)
            noisy_hist_img=drawHistogram(noisy_hist,img_zoom=3,shift_percent=1)
            noisy_hist_img=noisy_hist_img[noisy_hist_img.shape[0]//2:,:]

            hist_img=cv2.merge([hist_img,hist_img,noisy_hist_img]) # meh
            hist_img=cv2.line(hist_img,(240,0),(240,hist_img.shape[0]),(255,0,255),thickness=1)
            cv2.imshow('GROUND AND SKY H HISTs',hist_img)
            # TODO: highlight max, argmax in hist, sky_to_ground to ratio
            # cv2.addText(...)


    def splitCamera(self,ratio,hud=False):
        """
        splits supposed sky and ground segments of image into separate images
            division height-wise (full width) depending of ratio for speeding up
            the img. proc. pipeline
        :param ratio: ratio between the 2 parts
        :param hud: adds enclosing rectangles of sky, ground regions to self.dnoise_img
        """
        image=self.dnoise_img
        self.sky_image=image[:max(1,int(self.current_resolution[0]*ratio)),:]
        self.ground_image=image[int(self.current_resolution[0]*ratio)+1:,:]
        if hud:
            pt2= (self.current_resolution[1],int(self.current_resolution[0]*ratio))
            cv2.rectangle(image,(0,0),pt2,(0,0,255),thickness=3)
            cv2.rectangle(image,(0,pt2[1]+1), tuple(reversed(self.current_resolution)),(0,255,0),thickness=3)


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   BACKGROUND-FOREGROUND SEPARTION #   #   #   #   #   #
    # def preMask(self):
    #TODO
    # since some elements are known in color, it could be possible to use a mask
    # before thresholding to "save" some regions and avoid elimination; this could
    # help, simplyfing (ground) mask refinement and prevent loss of areas of interest
    # i.e. sky (100% removed), visual markers (0% removed)


    def otsuFilter(self,show_result=False,winname_prefix='',morph_ops=' '):
        """
        :param winname_prefix: str prefix of the window title
        :param morph_ops: (multi-char) list of ORDERED (left->right) morphological ops
        :param show_result: displays image of result
        PROS:   fast (13ms w/ half-sampled image)
                good results on binodal hist (clear separation sky-ground)

        CONS:   "artifacts" when unimodal hist (i.e. only ground)
                 removes optical markers
                 does not remove ice
                 not tune-able

        TODO: fast computation using between-class var approach
        """
        image = self.dnoise_img
        # manage h circularity
        h_shift=27
        hsv_image=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2HSV)
        h_image=hsv_image[...,0]
        h_image=(h_image+h_shift)%180
        _, otsu_mask=cv2.threshold(h_image,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        otsu_mask = self.maskRefinement(otsu_mask,morph_ops=morph_ops)
        self.otsu_mask = cv2.bitwise_not(otsu_mask)
        self.res_otsu=cv2.bitwise_and(image,image,mask=self.otsu_mask)
        if show_result:
            cv2.imshow(winname_prefix+' ground otsu',cv2.cvtColor(self.res_otsu,cv2.COLOR_RGB2BGR))


    # -- UNUSED --
    def adaptiveFilter(self,k_size=5,show_result=False,winname_prefix=''):
        """
        PROS:   ???
        CONS:   !!!
        """
        image = self.dnoise_img
        h_shift = 27
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
        h_image = hsv_image[..., 0]
        h_image = (h_image + h_shift) % 180
        ada_mask = cv2.adaptiveThreshold(h_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, k_size, 2)
        self.ada_mask = cv2.bitwise_not(ada_mask)
        self.res_ada = cv2.bitwise_and(image, image, mask=self.ada_mask)
        if show_result:
            cv2.imshow(winname_prefix+' ground ada', cv2.cvtColor(self.res_ada, cv2.COLOR_RGB2BGR))


    def balancedFilter(self,show_result=False,winname_prefix='',morph_ops=' '):
        """
        "iterative weight balancement" algorithm for finding histogram's barycenter
            to use it as a threshold
        PROS:   works great even in absence of sky
                removes ice better

        CONS:   a bit slower (17 ms)
                strange flickering (mask is totally ON/OFF) when observing one end
                    of the crater (far from ice, near fold aside the passage)
                    when no sky in frame
                        [even otsu goes nuts in that circumstance]
                removes visual targets
        """
        image = self.dnoise_img
        h_shift=27
        hsv_image=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2HSV)
        h_image=hsv_image[...,0]
        h_image=(h_image+h_shift)%180
        h_hist=np.histogram(h_image,180,[0,180])
        bal_tresh=self.balanced_hist_thresholding(h_hist)
        _, bal_mask=cv2.threshold(h_image,bal_tresh,255,cv2.THRESH_BINARY_INV)
        bal_mask = self.maskRefinement(bal_mask,morph_ops=morph_ops)
        self.bal_mask=cv2.bitwise_not(bal_mask)
        self.res_bal=cv2.bitwise_and(image,image,mask=self.bal_mask)
        if show_result:
            cv2.imshow(winname_prefix+' ground bal',cv2.cvtColor(self.res_bal,cv2.COLOR_RGB2BGR))
    @staticmethod
    def balanced_hist_thresholding(hist):
        left_end_pt = np.min(np.where(hist[0] > 0))
        right_end_pt = np.max(np.where(hist[0] > 0))
        barycenter_pt = (left_end_pt + right_end_pt) // 2
        left_force = np.sum(hist[0][0:barycenter_pt + 1])
        right_force = np.sum(hist[0][barycenter_pt + 1:right_end_pt + 1])
        while left_end_pt != right_end_pt:
            if right_force > left_force:
                # Removes the end weight
                right_force -= hist[0][right_end_pt]
                right_end_pt -= 1
                if ((left_end_pt + right_end_pt) // 2) < barycenter_pt:
                    left_force -= hist[0][barycenter_pt]
                    right_force += hist[0][barycenter_pt]
                    barycenter_pt -= 1
            else:
                left_force -= hist[0][left_end_pt]
                left_end_pt += 1
                if ((left_end_pt + right_end_pt) // 2) >= barycenter_pt:
                    left_force += hist[0][barycenter_pt + 1]
                    right_force -= hist[0][barycenter_pt + 1]
                    barycenter_pt += 1
        return barycenter_pt


    def rangeFilter(self,do_shift=False,show_result=False,winname_prefix='',morph_ops=' '):
        """
        :param do_shift: toggles shifting toward right (higher values) of H channel
                        hence avoiding circularity issues
        :param show_result: displays image of result
        :param morph_ops: ordered string of morphological operations to perform
        :param winname_prefix: prefix for window name (when showing result)
        simple channel range threshold; masked when all conditions sussist

        PROS:   does not remove visual targets
                precise, in the sense that only targets ground (reddish areas)
                no "glitching" when histogram not bimodal

        CONS:   hardcoded

        ALTERNATIVE APPROACH: USE LAB COLORSPACE
        + device independent (L might be a little)

        {YET: +hs mostrly condition indep
         BUT   -h cyclic, hence difference indoor/outdoor light possible) }
          L: dark(0)/bright(100)              ^+B/
                                              | / \ HUE ANGLE
          A: green(<0)/red(>0)        -A<----------->+A
          B: blue(<0)/yellow(>0)              |-B
        NOTE THAT HUE= ANGLE(from +A axis, ccw)
        """
        #TODO? could use sky_to_ground ratio to toggle which method to use for thresholding?

        #TODO: avoid red circularity by inverting the image
        #   and removing cyan {hsv:(90 - 10, 70, 50), (100, 255, 255)}
        image = self.dnoise_img
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if do_shift:
            red_max = (30, 255, 255)
            red_min = (0, 35, 50)
            range_mask = cv2.inRange(hsv_image, red_min, red_max)
            winname_prefix=winname_prefix+' shifted'
        else:
            red_max = (10, 255, 255)
            red_zeromax = (0, 30, 30)  # S=70
            red_zeromin = (180, 255, 255)
            red_min = (165, 30, 50)
            red_mask_min = cv2.inRange(hsv_image, red_zeromax, red_max)
            red_mask_max = cv2.inRange(hsv_image, red_min, red_zeromin)
            range_mask = cv2.bitwise_or(red_mask_min, red_mask_max)
            winname_prefix=winname_prefix+' '
        range_mask = self.maskRefinement(range_mask,morph_ops=morph_ops)
        self.range_mask=cv2.bitwise_not(range_mask)
        self.res_range = cv2.bitwise_and(image, image, mask=self.range_mask)
        if show_result:
            cv2.imshow(winname_prefix+' ground range',cv2.cvtColor(self.res_range,cv2.COLOR_RGB2BGR))


    def addNewSample(self):
        """
        manual sample selection from image from a static window
        test of correlation w/ while image not activated (prevents remotion of PARTICULARS)

        TODO: since selection occours over a single frame of video, it's prone to noise:
         should select a TEMPORAL MEAN of that region (but rover must stand still)

        TODO: no tested methid permitted to have the program working while selecting:
         not putting this in a different callback (freezes), spawning another listener,...
         ALSO: multithreading does not really exist thanks to GIL
         BUT: nogil

        TODO: multiple ROIs selection at once with cv2.selectROIs()

        TODO: comparison between histaograms of samples and whole image/dominant colors
            will be required for automatic selection of samples
         histogram comparison methods and relative tested data (static setup)
         hist_compare_dict = {
            "Correlation", (cv2.HISTCMP_CORREL, 0.2),  # or  0.4
            "Chi-Squared", (cv2.HISTCMP_CHISQR, 10000000),
            "Intersection", (cv2.HISTCMP_INTERSECT, 0.45),
            "Hellinger", (cv2.HISTCMP_BHATTACHARYYA, 0),
            "Kullback-Leibler", (cv2.HISTCMP_KL_DIV, 500)}
         _, compare_method, COMPARE_THRES = hist_compare_dict["Correlation"]
         image=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
         _,_,image_hist=computeHistogram(image,'hs',range_normalize=True)
         image_hist=image_hist.astype('float32')
         if cv2.compareHist(image_hist,sample_hist,hist_compare_method)>hist_compare_thres or bypass_comp:
        """
        winname="MOUSE+ENTER for selection,ESC to exit"
        img=self.dnoise_img
        rois=cv2.selectROIs(winname,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        # roi_x0,roi_y0,roi_x,roi_y=cv2.selectROI('SELECT NEW SAMPLE',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cv2.destroyWindow(winname)
        for (roi_x0,roi_y0,roi_x,roi_y) in rois:
            try:
                new_sampl=img[roi_y0:roi_y+roi_y0,roi_x0:roi_x+roi_x0]
                new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_RGB2HSV)
                _,_,sample_hist=computeHistogram(new_sample,'hs',range_normalize=True)
                sample_hist=sample_hist.astype('float32')
                self.samples.append(sample_hist)
                filename='../media/samples/sample_{}.jpg'.format(str(time.time()))
                cv2.imwrite(filename,cv2.cvtColor(new_sampl,cv2.COLOR_RGB2BGR))
            except Exception as exc:
                print('sample aquisition aborted '+str(exc)[13])
        print("number of current samples: {}".format(len(self.samples)))


    def removeSampleCallback(self,index_to_remove_msg):
        """
        removes a sample by its index from the sampleFilter array
        :param index_to_remove_msg: integer, val<0 counts from the end
        TODO? does this work with empty msg? technically .pop()==.pop(-1)
        """
        try:
            self.samples.pop(index_to_remove_msg)
        except:
            print('could not remove that element')


    def sampleFilter(self,show_result=False, winname_prefix='', morph_ops=' '):
        """
        filter camera stream by histogram backprojection of selected samples
         builds a probability map of a certain pixel of the target img to be part of
         the sample; a treshold defines the binary mask

        :param show_result: displays image of result
        :param morph_ops: ordered string of morphological operations to perform
        :param winname_prefix: prefix for window name (when showing result)

        PROS:   adaptability
                no use of V channel (heavily condition/device dependent)
                samples can be remove if not necessary/ erroneous
                 (e.g. by checking disapparence of objects of interest)
                possible offline setup

        CONS:   slower than range filter
                overhead increases with samples amount
                hard to refine fue to iterative process

        TODO: automatic sample selection by dominant colors, image division and analysis,...

        TODO: adapt filter refinement, shape, kernel dimension wrt sample dimension
         relative data deprending on which sample retrieve method is used
         if retrieve_method == 'from storage':
            k_size = 3
            k_shape = cv2.MORPH_RECT
            mask_thresh = 10  # CORREL:10, CHISQR:0-20,
         elif 'live':
            # selects from current camera stream
            num_samples_sqrt = 4 #since will devide each axis of image this
            mask_h = self.sim_img.shape[0] // num_samples_sqrt
            mask_w = self.sim_img.shape[1] // num_samples_sqrt
            k_size = 4
            k_shape = cv2.MORPH_RECT
            mask_thresh = 50

        TODO: nothing to make sample selection non blocking worked as for now

        TODO? 3 channel mask any better? as for my tests, no
            (and it would have been counterintuitive)

        TODO: possibly best refinement would be a CLOSING w/ a SMALL kernel

        TODO ALSO: better refinement before or after the union of indivudual masks?

        TODO: init the mask outside (e.g. when computing self.current_resolution)
         to prevent if block and other issues

        TODO: cv2.calcBackProjection accepts an array of images: MUST BE SAME SIZE;
            could have a constant saple size using onClick events
            ELSE: could interpolate the images
        """
        image = self.dnoise_img
        if self.select_new_sample:
            self.select_new_sample = False
            self.pause_stats=True
            self.addNewSample()
        if self.samples:#==if not self.samples==[]
            hsv_image=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2HSV)
            _,_,image_hist=computeHistogram(hsv_image,'hs',range_normalize=True)
            sample_mask=np.zeros(np.asarray(self.current_resolution),dtype=np.uint8)

            for sample_hist in self.samples:
                add_mask=cv2.calcBackProject([hsv_image],[0,1],sample_hist,[0,180,0,256],1)

                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cv2.filter2D(add_mask, -1, kernel, add_mask)
                # add_mask = self.maskRefinement(add_mask, morph_ops=morph_ops)
                cv2.normalize(add_mask, add_mask, 0, 255, cv2.NORM_MINMAX)
                _, add_mask = cv2.threshold(add_mask, 50, 255, 0)

                sample_mask=cv2.bitwise_or(sample_mask,add_mask)
            self.sample_mask=cv2.bitwise_not(sample_mask)
            self.sample_mask = self.maskRefinement(self.sample_mask, morph_ops=morph_ops)
            self.res_sample = cv2.bitwise_and(image, image, mask=self.sample_mask)
        else:
            self.res_sample=image
        if show_result:
            cv2.imshow(winname_prefix + ' ground sample', cv2.cvtColor(self.res_sample, cv2.COLOR_RGB2BGR))


    @staticmethod
    def maskRefinement(mask,morph_ops=' ',k_size=5,k_shape=cv2.MORPH_ELLIPSE):
        """
        refine the mask w/ morphological operations

        :param mask: input mask to be refined
        :param morph_ops: ORDERED string of required morph. operations
        :param k_size: kernel characteristic size
        :param k_shape: kernel shape:cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE}

        in some previous test (white book over fuzzy carpet), best combinations (ordered by results):
            1) open (hitmiss (erode
            1) dilate (erode
            2) hitmiss (erode
            3) dilate (hitmiss
            4) dilate (open
            5) hitmiss
            6) dilate
            7) erode

        TODO: refine number of iterations based on image, samples,...
        """
        kernel = cv2.getStructuringElement(k_shape, (k_size, k_size))
        for mop in morph_ops:
            if mop=='d'or mop=='dilate':
                mask=cv2.dilate(mask.copy(),kernel,iterations=1)
                # NOTE: mask=cv2.morphologyEx(mask.copy(), cv2.MORPH_DILATE, kernel)
            elif mop=='e'or mop=='erode':
                mask=cv2.erode(mask.copy(),kernel,iterations=1)
            elif mop=='h'or mop=='hitmiss':
                mask=cv2.morphologyEx(mask.copy(), cv2.MORPH_HITMISS, kernel)
            elif mop=='o'or mop=='open':
                mask=cv2.morphologyEx(mask.copy(), cv2.MORPH_OPEN, kernel)
            elif mop=='c'or mop=='close':
                mask=cv2.morphologyEx(mask.copy(), cv2.MORPH_CLOSE, kernel)
        return mask


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   VISUALIZATION TOOLS #   #   #   #   #   #
    def showChannels(self,noisy='d',channels=' ',downsize_factor=1,winname_prefix='',h_shift=0):
        """
        Shows selected image channels
        :param noisy: toggles sim or noisy input
        :param channels: permitted 'hsvrgbGL' (cumulative string)
                h s v : HSV colorspace
                r g b : RGB colorspace
                GL    : GreyLevels
                f     : full rgb image
        :param downsize_factor: final dimensions will be devided by this value
            NOTE: intended to be a "cosmetics" tool for lower screen resolutions
        :param winname_prefix: window text prefix
        :param h_shift: shift the h channel value values toward rigth of the histogram
                        highlighting similarity of values around 0==180 (especially for noisy imgs)
        """
        if noisy == 'n' or noisy == 'noisy':
            image = self.noisy_img
            winname_prefix = winname_prefix + ' NOISY '
        elif noisy == 'd' or noisy == 'denoised':
            image = self.dnoise_img
            winname_prefix = winname_prefix + ' DENOISED '
        else:
            image = self.sim_img
        if downsize_factor != 1:
            new_size = (int(image.shape[1] / downsize_factor), int(image.shape[0] / downsize_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        if 'f' in channels:
            rgb_out = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
            cv2.imshow(winname_prefix + 'FULL RGB ', rgb_out)
        if 'r' in channels:
            cv2.imshow(winname_prefix + "R", image[:, :, 0])
        if 'g' in channels:
            cv2.imshow(winname_prefix + "G", image[:, :, 1])
        if 'b' in channels:
            cv2.imshow(winname_prefix + "B", image[:, :, 2])
        if 'h' in channels or 's' in channels or 'v' in channels:
            hsv_in = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
            if 'h' in channels:
                out = (hsv_in[:, :, 0] + h_shift) % 180
                cv2.imshow(winname_prefix + "H", out)
            if 's' in channels:
                cv2.imshow(winname_prefix + "S", hsv_in[:, :, 1])
            if 'v' in channels:
                cv2.imshow(winname_prefix + "V", hsv_in[:, :, 2])
        if 'GL' in channels:
            gl_out = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
            cv2.imshow(winname_prefix + "GL", gl_out)


    def showHistograms(self,channels,noisy='d',winname_prefix=''):
        """
        computes and shows H,S and RGB histograms
        :param noisy: toggles sim or noisy input
        :param channels: which channel to analyze and show
        :param winname_prefix: window text prefix
        :internal:  xyz_scaling: img height rescaling,
                    zyx_shift: shift toward right side of hist (higher values)

        TODO: optional superimposing of noisy histogram
        """
        rgb_scaling = 0.01
        hsv_shift = 15
        rgb_shift=92
        if noisy == 'n' or noisy == 'noisy':
            winname_prefix = winname_prefix + ' NOISY '
            rgb_in_image = self.noisy_img
        elif noisy == 'd' or noisy == 'denoised':
            rgb_in_image = self.dnoise_img
            winname_prefix = winname_prefix + ' DENOISED '
        else:
            rgb_in_image = self.sim_img
        shift_text=", SHIFT AMOUNT= " + str(int(rgb_shift*2.56)) + " /256"
        if 'r' in channels or 'f'in channels:
            r_hist, _ = computeHistogram(rgb_in_image, 'r')
            r_hist_img=drawHistogram(r_hist,img_zoom=3,shift_percent=rgb_shift,man_scale=rgb_scaling)
            if 'f'not in channels:
                cv2.imshow(winname_prefix + "R"+shift_text, r_hist_img)
        if 'g' in channels or'f'in channels:
            g_hist, _ = computeHistogram(rgb_in_image, 'g')
            g_hist_img=drawHistogram(g_hist,img_zoom=3,shift_percent=rgb_shift,man_scale=rgb_scaling)
            if 'f' not in channels:
                cv2.imshow(winname_prefix + "G", g_hist_img)
        if 'b' in channels or'f'in channels:
            b_hist, _ = computeHistogram(rgb_in_image, 'b')
            b_hist_img=drawHistogram(b_hist,img_zoom=3,shift_percent=rgb_shift,man_scale=rgb_scaling)
            if 'f'not in channels:
                cv2.imshow(winname_prefix + "B", b_hist_img)
        if 'f'in channels:
            rgb_hist_img = cv2.merge([b_hist_img, g_hist_img, r_hist_img])
            cv2.imshow(winname_prefix+"RGB HIST"+shift_text, rgb_hist_img)

        if 'h' in channels or 's' in channels or 'v' in channels:
            shift_text = ", SHIFT AMOUNT= " + str(int(hsv_shift * 1.8)) + " /180"
            hsv_in_image = cv2.cvtColor(rgb_in_image.copy(), cv2.COLOR_RGB2HSV)
            if 'h' in channels:
                _, _, h_hist = computeHistogram(hsv_in_image, 'h', range_normalize=True)
                h_hist_img = drawHistogram(h_hist, img_zoom=3, shift_percent=hsv_shift)
                h_hist_img=h_hist_img[h_hist_img.shape[0]//2:,:]
                cv2.imshow(winname_prefix+"H"+shift_text,h_hist_img)
            if 's' in channels:
                _, _, s_hist = computeHistogram(hsv_in_image, 's', range_normalize=True)
                s_hist_img= drawHistogram(s_hist,img_zoom=3)
                s_hist_img=s_hist_img[s_hist_img.shape[0]//2:,:]
                cv2.imshow(winname_prefix+"S"+" NOT SHIFTED",s_hist_img)
            if 'v' in channels:
                _, _, v_hist = computeHistogram(hsv_in_image, 'v', range_normalize=True)
                v_hist_img= drawHistogram(v_hist,img_zoom=3)
                v_hist_img=v_hist_img[v_hist_img.shape[0]//2:,:]
                cv2.imshow(winname_prefix+"V"+" NOT SHIFTED",v_hist_img)
        if 'GL' in channels:
            gl_out = cv2.cvtColor(rgb_in_image, cv2.COLOR_RGB2GRAY)
            cv2.imshow(winname_prefix + "GL", gl_out)


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
if __name__ == '__main__':
    node_name = 'image_filter'
    rospy.init_node(node_name, anonymous=False)
    print("\nNavCam Image Prepocessing Node for Navigation Pipeline")
    print('node name: ' + node_name)

    preprocessor = ImagePreprocNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit(0)
    rospy.loginfo("exiting...")
    # cv2.destroyAllWindows()

##########===================##################
#TODO: FAST K CLUSTERING
# https://stackoverflow.com/questions/51176438/opencv-determine-what-two-hue-values-would-best-represent-an-image
# https://www.timpoulsen.com/2018/finding-the-dominant-colors-of-an-image.html

#TODO: DOMINANT COLORS OF IMAGE
# https://www.aishack.in/tutorials/dominant-color/

#TODO: faster approach using between-class variances
# https://theailearner.com/2019/07/19/optimum-global-thresholding-using-otsus-method/

"""
BIBLIOGRAFIA

COLORE MEDIO vs COLORI DOMINANTI
    MEDIO:          avg_color = numpy.average(numpy.average(image, axis=0), axis=0)
    
    DOMINANTI:      autovettori
                    k clustering -> slow (~200ms cycle)


FILTRI

       if symmetric kernel -> correlation
        cv2.filter2D(image, -1, kernel, anchor)
        anchor lies within kernel (default: (-1,-1) kernel center)

        if not symm-> convolution
        YET still use correlation WITH FLIPPED KERNEL
        cv2.flip(kernel) #opencv does not automatically flip kernel
        flipped_anchor=(kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1)


ULTERIORI MIGLIORAMENTI

-usare DEPTH+COLORE x l'analisi del terreno: x esempio i sassi sono abbastanza riconoscibili dallo
 sfondo nell'immagine di profondità, con un'opportuna thres si potrebbero individuare
"""




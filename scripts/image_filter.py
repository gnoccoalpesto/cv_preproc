#!/usr/bin/env python3
"""
AUTHOR:     FRANCESCO CERRI
            francesco.cerri2@studio.unibo.it

            ALMA-X TEAM
            UNIVERSITÀ DEGLI STUDI DI BOLOGNA ALMA MATER STUDIORUM

THIS CODE REMOVES THE TERRAIN (AND OTHER SIMILAR FEATURES, POSSIBLY)
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
# histograms analysis tools definitions
from lib_histograms import computeHistogram, drawHistogram
#TODO: fix import
#TODO? ALSO import class methods?
import time
from glob import glob


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
class GroundFilter:
    """    """
    def __init__(self):
        #TODO: use the other camera;
        # wheels remotion can be active by reading:
        # /joint_stats.rocker_{L,R}_joint
        camera_dict = { 'NavCam':   "/camera/image_raw",
                        'HazCam':   "zed2/left_raw/image_raw_color",
                        'nsyHazCam':"/camera/image_raw/noisy",
                        'prprocHazCam':"/camera/image_raw/preproc"}

        self.input_topic = rospy.get_param(
            '/image_preproc/preproc_camera_topic',camera_dict['prprocHazCam'])

        self.filtered_topic = "/camera/image_raw/filtered"
        print('listened topic: ' + self.input_topic)
        print('published topic: ' + self.filtered_topic)

        self.preproc_img = np.ndarray
        self.noisy_img = np.ndarray
        self.in_dtype = None
        self.camera_channels=int
        self.current_resolution= tuple
        self.print_once_resolution=True
        self.filtered_img=np.ndarray

        self.cvbridge = CvBridge()

        # TODO: add rosparams to change stuff at runtime

        #TODO: "param"_dict={'morph_param':'/image_preproc/morph_param',
        #                     'morph_topic':...}
        self.morph_ops_source='/image_preproc/morph_ops'
        self.add_sample_source='image_preproc/add_request'
        # depth data already subsampled of same amount
        self.depth_topic='/zed2/depth/depth_preproc'
        # / zed2 / depth / depth_registered
        # imu_dict = {'imu': "/imu/data_raw",
        #             'camera': "/zed2/imu/data"}
        # self.imu_topic=imu_dict['imu']
        # self.imu_msg = []

        # obtains a ~10ms cycle len, considering a bit of superimposed overhead
        FILTER_BASE_FREQ=105
        filter_freq=rospy.get_param('image_preproc/ground_filter_freq',FILTER_BASE_FREQ)
        filter_cycle_time_nsec=rospy.Duration(nsecs=int(1E9/filter_freq))
        # input_cycle_time_nsec=filter_cycle_time_nsec/5

        self.sky_to_groud=float
        self.sky_image=np.ndarray
        self.ground_image=np.ndarray
        self.sky_mask=np.ndarray

        self.MORPH_OPS='1'#=='c'

        self.MENU_IMAGE=np.ndarray
        self.res_sample= np.ndarray
        self.res_range= np.ndarray
        self.sample_mask= np.ndarray
        self.range_mask= np.ndarray
        self.select_new_sample=False
        self.samples=[]
        self.averaging_samples=[]
        self.sample_mask=None
        self.sample_source='/sim_ws/src/almax/cv_preproc/media/samples/'
        self.addedAllSample=False

        self.SHOW_RESULT=False
        self.selected_premask='c'
        # available premasks:   c   color
        #                       d   depth
        self.selected_filter='r'
        # available filters:    s   sample
        #                       r   range
        self.toggle_sample=False
        self.toggle_range=True
        self.toggle_morph=True

        self.depth_mask=np.ndarray

        #   PERFORMANCES   #   #   #   #   #   #
        self.pause_stats=False
        self.iter_stamp=0
        self.iter_counter=0
        self.ave_time=0

        #   KCLUSTERS     #   #   #   #   #   #
        self.n_colors = 12
        self.cl_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.cl_flags = cv2.KMEANS_RANDOM_CENTERS

        #   #   #   #   #   #   #   #   #
        self.initFilter()

        #   ROS I/O OPERATIONS   #   #   ##   #   #   #   #   ##   #   #   #   #   #
        self.filteredPublisher = rospy.Publisher(self.filtered_topic, Image, queue_size=1)
        #TODO: timer does not solve blocking wait for cv2.selectROIs
        # ALSO, no real gain controlling input sampling rate manually
        # (BUT possibly unregistering overhead w/ rospy.wait_for_message(): avoid)
        # self.inputTimer = rospy.Timer(input_cycle_time_nsec, self.inputTimerCallback)
        self.inputListener = rospy.Subscriber(self.input_topic, Image, self.inputCallback)

        self.morphOpsListener = rospy.Subscriber(self.morph_ops_source, String, self.morphOpsCallback)

        self.addRequestListener = rospy.Subscriber(self.add_sample_source, Bool, self.addSampleCallback)
        # TODO: for remotion requests; could modify the one above
        # self.removeRequestListener=...

        # self.imuListener=rospy.Subscriber(self.imu_topic,Imu,self.imuCallback)
        self.depthListener=rospy.Subscriber(self.depth_topic,Image,self.depthCallback)
        time.sleep(.25)
        self.noisyListener = rospy.Subscriber('/camera/image_raw/noisy', Image, self.noisyCallback)
        self.this_time=time.time()
        # overhead can be controlled by setting a different cycle frequency
        self.filterTimer=rospy.Timer(filter_cycle_time_nsec,self.filterCallback)



    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   INITIALIZATION, I/O CALLBACKS, STATISTICS   #   #   #   #   #   #
    def initFilter(self):
        print('initializing...')
        img_msg = rospy.wait_for_message(self.input_topic, Image)
        image = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        self.preproc_img=image
        # original dtype is uint8, display also accepts float32 (NOTE: result uint8==float32/255)
        self.in_dtype = image.dtype
        self.camera_channels=image.shape[2]
        self.current_resolution = image.shape[:2]
        self.sky_mask=np.zeros_like(image)
        self.depth_mask=np.zeros_like(image)
        self.MENU_IMAGE=np.zeros_like(image)
        MENU_TEXT="Esc: exit (RESPAWNS: Ctrl+C in terminal)\ns: displays sample filter\nr: displays range filter\n"
        MENU_TEXT=MENU_TEXT+"z: enables all samples for sample filter\nx: disables all samples\n"
        MENU_TEXT=MENU_TEXT+"l: add new sample(s) to filter from ROI(s) selection;\n      multiple allowed, Esc to stop\n"
        MENU_TEXT=MENU_TEXT+"k: same as 'l' & new samples are saved\nm: increase morphological operations amount\n"
        MENU_TEXT=MENU_TEXT+"n: decrease morphological operations amount\nd: select depth based (sky) prefilter\n"
        MENU_TEXT = MENU_TEXT +"c: select color based (sky) prefilter\n"
        MENU_TEXT=MENU_TEXT+"\nSAMPLE FOLDER: ../media/samples/"
        y0, dy = 18, cv2.getTextSize(MENU_TEXT,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][1]
        # text splitting and multi line printing
        for ii, line in enumerate(MENU_TEXT.split('\n')):
            y = y0 + ii * int(1.2*dy)
            cv2.putText(self.MENU_IMAGE,line,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255))
        print("INPUT\t==\t==\t==\n resolution: {}\n channels: {}\n depth type: {}".
              format(str(self.current_resolution),str(self.camera_channels),str(self.in_dtype)))


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


    #--UNUSED--
    # def imuCallback(self,imu_msg):
    # TODO: refine sky recognition using cross imu data:
    #  roll angle sets inclination of division line (horizon tilting)
    #  pitch angle sets distance from the upper corner:
    #   >0 toward x center (width wise direction) (horizon up)
    #   <0 toward y center (height wise direction) == far from x center (horizon down)
    # https://wiki.ros.org/imu_filter_madgwick
    #     self.imu_msg=imu_msg


    def depthCallback(self,depth_msg):
        """
        acquires depth camera information
        then computes a mask based on a certaint threshold (nan: threshold=max sensor range)

        TODO: modify to implement variable distance thresholding; could be usefull
            to focus attention on close areas;
                .astype('float32') gives values decreasing with proximity (min ~0.4)
                BUT still nan outside usefull range
        """
        try:
            depth_data = self.cvbridge.imgmsg_to_cv2(depth_msg,depth_msg.encoding)
            self.depth_mask=np.zeros(self.current_resolution).astype('uint8')
            self.depth_mask[np.isnan(depth_data)]=255
        except CvBridgeError:
            # TODO: self.bridgerror
            print("DEPTH: cv bridge error")
        except : pass


    def inputCallback(self,img_msg):
        """
        'Real Time' (threaded) ascquisition of preprocessed input image
        """
        try:
            self.preproc_img = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        except CvBridgeError:
            # TODO: self.bridgerror
            print("INPUT: cv bridge error")


    def noisyCallback(self, img_msg):
        """
        'Real Time' (threaded) ascquisition of noisy input image
        """
        try:
            self.noisy_img = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        except CvBridgeError:
            print("noisy: cv bridge error")


    def inputTimerCallback(self,_):
        """
        same as above, but using specific rospy.Timer insterad of incoming message to
            spawn this callback

        NOTE: had to manually add an exception handling in:
        /opt/ros/noetic/lib/python3/dist-packages/rospy/impl/tcpros_base.py
        added rows: 859(except AttributeError)...865(pass)
        """
        try:
            img_msg=rospy.wait_for_message(self.input_topic, Image)
            self.preproc_img = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        except CvBridgeError:
            print("cv bridge error")


#    ##################################################################################
###    ##################################################################################
######    ##################################################################################

    # MAIN LOOP : every time a msg on self.input_topic received
    def filterCallback(self, _):
        self.updateStatistics(self.this_time)
        # self.showHistograms('h')
        # self.showHistograms('h',noisy='n')

        # IMAGE ANALYSIS --------------------------------------------------------------------
        # self.cameraAnalysis(show_hist=True)
        # self.splitCamera(self.sky_to_groud,hud=False)
        # self.cameraClustering()

        # BACKGROUND FILTERING -------------------------------------------------------------------------
        morph_ops = self.MORPH_OPS# if self.toggle_morph else ''

        if not self.SHOW_RESULT:
            cv2.imshow("keystrokes menu",self.MENU_IMAGE)

        self.preMask(sky_mask_method=self.selected_premask)
        if self.toggle_sample:
            self.sampleFilter(show_result=self.SHOW_RESULT, morph_ops=morph_ops)
            self.filtered_img=self.res_sample.copy()
        elif self.toggle_range:
            self.rangeFilter(show_result=self.SHOW_RESULT, morph_ops=morph_ops)
            self.filtered_img=self.res_range.copy()

        # OUTPUT ------------------------------------------------------------------------------------
        #TODO: superimpose time matermark on output to understand when output stopped

        # self.showHistograms(noisy='p',channels='h')
        # self.showChannels(channels='f')
        # self.showChannels(channels='hsv',h_shift=30)

        out_img=self.filtered_img
        # out_img=cv2.cvtColor(self.filtered_img,cv2.COLOR_RGB2BGR)
        out_msg = self.cvbridge.cv2_to_imgmsg(out_img)
        self.filteredPublisher.publish(out_msg)

        ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
        # TODO: cv2.pollKey() better because saves few msecs?
        # k = cv2.waitKey(1) & 0xFF
        k = cv2.pollKey() & 0xFF
        if k == 27:#Esc
            rospy.signal_shutdown('Esc key pressed')
        elif k!=255:#NO KEY PRESSED
            # print(k)
            self.keyAction(k)

######    ##################################################################################
###    ##################################################################################
#    ##################################################################################

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   IMAGE PREPROCESSING
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
        image = self.preproc_img
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
        image = self.preproc_img
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
            hist_img=cv2.line(hist_img,(240,0),(240,hist_img.shape[0]),(255,0,255),thickness=1)
            cv2.imshow('GROUND AND SKY H HISTs',hist_img)
            # TODO: highlight max, argmax in hist, sky_to_ground to ratio
            # cv2.addText(...)


    def splitCamera(self,ratio,hud=False):
        """
        splits supposed sky and ground segments of image into separate images
            division height-wise (full width) depending on ratio for speeding up
            the img. proc. pipeline
        :param ratio: ratio between the 2 parts
        :param hud: adds enclosing rectangles of sky, ground regions to self.preproc_img
        """
        image=self.preproc_img
        self.sky_image=image[:max(1,int(self.current_resolution[0]*ratio)),:]
        self.ground_image=image[int(self.current_resolution[0]*ratio)+1:,:]
        if hud:
            pt2= (self.current_resolution[1],int(self.current_resolution[0]*ratio))
            cv2.rectangle(image,(0,0),pt2,(0,0,255),thickness=3)
            cv2.rectangle(image,(0,pt2[1]+1), tuple(reversed(self.current_resolution)),(0,255,0),thickness=3)


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   BACKGROUND-FOREGROUND SEPARTION
    def preMask(self,sky_mask_method='color'):
        """
        saves a bit of computation by masking some obkjects which are known for sure
         ie. : sky

        :param sky_mask_method: which method uses for removing sky from image
                    available methods:
                    -"color","c": uses blue color range for remotion
                        PRO:    depth sensor's max range independent
                        CON:    camera & condition dependent

                    -"depth","d": uses depth channel
                        PRO&CON:    viceversa as above
                        +CON:       also for objects w/ distance <MIN_SENSOR_RANGE

        TODO: implement a mask for objects CERTAINTLY NOT TO REMOVE FROM IMAGE
            (i.e. markers, probes, rocks,...)

        TODO? mask refinement could be done here too( hence twice)?

        TODO: causes some "holes" in white objects

        TODO: wheel remotion when using frame camera
        """
        # if self.input_topic=='FRAME_CAMERA_INPUT_TOPIC'
        #     draw_rectangles_where_wheel_is()
        if 'c'in sky_mask_method or sky_mask_method=='color':
            image = self.preproc_img
            # hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
            hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
            #sky filtering
            blue_min=(105, 0, 0)
            blue_max=(135, 255, 255)
            self.sky_mask=cv2.inRange(hsv_image,blue_min,blue_max)
        elif 'd'in sky_mask_method or sky_mask_method=='depth':
            self.sky_mask=self.depth_mask


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
        image = self.preproc_img
        # manage h circularity
        h_shift=27
        # hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        h_image=hsv_image[...,0]
        h_image=(h_image+h_shift)%180
        _, otsu_mask=cv2.threshold(h_image,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        otsu_mask=cv2.bitwise_or(otsu_mask,self.sky_mask)
        otsu_mask = cv2.bitwise_not(otsu_mask)
        ref_otsu_mask = self.maskRefinement(otsu_mask,morph_ops=morph_ops)
        res_otsu=cv2.bitwise_and(image,image,mask=ref_otsu_mask)
        if show_result:
            cv2.imshow(winname_prefix+' OTSU FILTER RESULT',res_otsu)
            # cv2.imshow(winname_prefix+' OTSU FILTER RESULT',cv2.cvtColor(res_otsu,cv2.COLOR_RGB2BGR))


    # -- UNUSED --
    def adaptiveFilter(self,k_size=5,show_result=False,winname_prefix=''):
        """
        PROS:   none?
        CONS:   !!!
        """
        image = self.preproc_img
        h_shift = 27
        # hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        h_image = hsv_image[..., 0]
        h_image = (h_image + h_shift) % 180
        ada_mask = cv2.adaptiveThreshold(h_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, k_size, 2)
        ada_mask=cv2.bitwise_or(ada_mask,self.sky_mask)
        ada_mask = cv2.bitwise_not(ada_mask)
        res_ada = cv2.bitwise_and(image, image, mask=ada_mask)
        if show_result:
            cv2.imshow(winname_prefix+' ADAPTIVE FILTER RESULT', res_ada)
            # cv2.imshow(winname_prefix+' ADAPTIVE FILTER RESULT', cv2.cvtColor(res_ada, cv2.COLOR_RGB2BGR))


    def balancedFilter(self,show_result=False,winname_prefix='',morph_ops=' '):
        """
        'iterative weight balancement' algorithm for finding histogram's barycenter
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
        image = self.preproc_img
        h_shift=27
        # hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        h_image=hsv_image[...,0]
        h_image=(h_image+h_shift)%180
        h_hist=np.histogram(h_image,180,[0,180])
        bal_tresh=self.balanced_hist_thresholding(h_hist)
        _, bal_mask=cv2.threshold(h_image,bal_tresh,255,cv2.THRESH_BINARY_INV)
        bal_mask = cv2.bitwise_or(bal_mask, self.sky_mask)
        bal_mask = cv2.bitwise_not(bal_mask)
        ref_bal_mask = self.maskRefinement(bal_mask,morph_ops=morph_ops)
        res_bal=cv2.bitwise_and(image,image,mask=ref_bal_mask)

        if show_result:
            cv2.imshow(winname_prefix+' BALANCED FILTER RESULT',res_bal)
            # cv2.imshow(winname_prefix+' BALANCED FILTER RESULT',cv2.cvtColor(res_bal,cv2.COLOR_RGB2BGR))

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
        simple channel range threshold; masked when all conditions sussist

        :param do_shift: toggles shifting toward right (higher values) of H channel
                        hence avoiding circularity issues
        :param show_result: displays image of result
        :param morph_ops: ordered string of morphological operations to perform
        :param winname_prefix: prefix for window name (when showing result)

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
        TODO? could use sky_to_ground ratio to toggle which method to use for thresholding?

        TODO: avoid red circularity by inverting the image
          and removing cyan {hsv:(90 - 10, 70, 50), (100, 255, 255)}

        TODO: apply self.sky_mask before to compute here to reduce computation amount
            apply contextually an "ignore black pxs mask" to ignore already masked regions
            (and also visual markers)
        """
        image = self.preproc_img.copy()
        # hsv_image=np.zeros_like(image)
        # image[np.where(~self.sky_mask)]=0
        # cv2.findNonZero()
        # hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        if do_shift:
            red_max = (30, 255, 255)
            red_min = (0, 35, 50)
            red_mask = cv2.inRange(hsv_image, red_min, red_max)
            winname_prefix=winname_prefix+' shifted'
        else:
            red_max = (15, 255, 255)
            red_zeromax = (0, 30, 30)  # S=70
            red_zeromin = (180, 255, 255)
            red_min = (165, 30, 50)
            red_mask_min = cv2.inRange(hsv_image, red_zeromax, red_max)
            red_mask_max = cv2.inRange(hsv_image, red_min, red_zeromin)
            red_mask = cv2.bitwise_or(red_mask_min, red_mask_max)
            winname_prefix=winname_prefix+' '
        range_mask = cv2.bitwise_or(red_mask, self.sky_mask)
        range_mask = cv2.bitwise_not(range_mask)
        self.range_mask = self.maskRefinement(range_mask,morph_ops=morph_ops)
        self.res_range = cv2.bitwise_and(image, image, mask=self.range_mask)
        if show_result:
            cv2.imshow(winname_prefix+' RANGE FILTER RESULT',self.res_range)
            # cv2.imshow(winname_prefix+' RANGE FILTER RESULT',cv2.cvtColor(self.res_range,cv2.COLOR_RGB2BGR))


    def distanceFilter(self,dist_method='mahalanobis',show_result=False,winname_prefix='',morph_ops=' '):
        """
        filtering based on color distance;
        improved using Einstein summation convention that avoids loops which are extremely slow

        :param dist_method: method for distance computation; available:
                                mahalanobis: weighted on variance
                                euclidean: simple distance in colorspace
        :param show_result: displays image of result
        :param morph_ops: ordered string of morphological operations to perform
        :param winname_prefix: prefix for window name (when showing result)

        PROS:
        CONS:       slow

        TODO: use a real device-independent colorspace

        TODO: apply self.sky_mask before to compute here to reduce computation amount
            apply contextually an "ignore black pxs mask" to ignore already masked regions
            (and also visual markers)

        https://github.com/tvanslyke/ImageProcessing/blob/master/Mahalanobis.py
        https://dsp.stackexchange.com/questions/1625/basic-hsb-skin-detection-neon-illumination

        """
        image = self.preproc_img
        threshold=3

        pre_distance_mask=self.myDistance(dist_method=dist_method)
        _,distance_mask=cv2.threshold(pre_distance_mask, threshold, 255,cv2.THRESH_BINARY_INV)
        distance_mask = cv2.bitwise_or(distance_mask.astype('uint8'), self.sky_mask)
        distance_mask = cv2.bitwise_not(distance_mask)
        ref_distance_mask = self.maskRefinement(distance_mask,morph_ops=morph_ops)
        res_distance = cv2.bitwise_and(image, image, mask=ref_distance_mask)
        if show_result:
            cv2.imshow(winname_prefix+' DISTANCE FILTER RESULT',res_distance)
            # cv2.imshow(winname_prefix+' DISTANCE FILTER RESULT',cv2.cvtColor(res_distance,cv2.COLOR_RGB2BGR))


    def myDistance(self, dist_method='mahalanobis',mean_pix=None):
        """
        Einstain summation convention distance computation
        :param mean_pix:    mean value to compute distance with;
                            if None provided, will use img's mean
                            ASSUMED COLUMN VECTOR
                            ASSUMED COLUMN VECTOR
        :param dist_method: method for distance computation; available:
                                mahaSlanobis: weighted on variance
                                euclidean: simple distance in colorspace
        """
        image = self.preproc_img
        arr = np.reshape(image,(self.current_resolution[0]*self.current_resolution[1],self.camera_channels))
        meandiff = arr - (mean_pix if mean_pix is not None else np.mean(arr, axis=0))
        if dist_method=='mahalanobis' or 'm' in dist_method:
            invcovar = np.linalg.inv(np.cov(np.transpose(arr)))
            output = np.dot(meandiff, invcovar)
            return np.sqrt(np.einsum('ij,ij->i', output, meandiff)).reshape(self.current_resolution)
        # dist_method=='euclidean'
        return np.sqrt(np.einsum('ij,ij->i',meandiff, meandiff)).reshape(self.current_resolution)


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

        TODO: apply self.sky_mask before to compute here to reduce computation amount
            apply contextually an "ignore black pxs mask" to ignore already masked regions
            (and also visual markers)
        """
        image = self.preproc_img
        if self.select_new_sample:
            self.select_new_sample = False
            self.pause_stats=True
            self.addNewSample()
        if self.samples:#==if not self.samples==[]
            # hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
            hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
            _,_,image_hist=computeHistogram(hsv_image,'hs',range_normalize=True)
            sample_mask=np.zeros(np.asarray(self.current_resolution),dtype=np.uint8)

            for sample_hist in self.samples:
                add_mask=cv2.calcBackProject([hsv_image],[0,1],sample_hist,[0,180,0,256],1)

                #TODO: improve refinement
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                #TODO: filter does not automatically flip the kernel (for convolution), yet it is symmetrical
                cv2.filter2D(add_mask.copy(), -1, kernel, add_mask)
                # add_mask = self.maskRefinement(add_mask, morph_ops=morph_ops)
                cv2.normalize(add_mask, add_mask, 0, 255, cv2.NORM_MINMAX)
                _, add_mask = cv2.threshold(add_mask, 50, 255, 0)

                sample_mask=cv2.bitwise_or(sample_mask,add_mask)
            sample_mask=cv2.bitwise_or(sample_mask,self.sky_mask)
            sample_mask=cv2.bitwise_not(sample_mask)
            self.sample_mask = self.maskRefinement(sample_mask, morph_ops=morph_ops)
        else:
            sample_mask = cv2.bitwise_not(self.sky_mask)
            self.sample_mask = self.maskRefinement(sample_mask, morph_ops=morph_ops)

        self.res_sample = cv2.bitwise_and(image, image, mask=self.sample_mask)

        if show_result:
            cv2.imshow(winname_prefix + ' SAMPLE FILTER RESULT, {} samples'.format(len(self.samples)),
                       self.res_sample)
                       # cv2.cvtColor(self.res_sample, cv2.COLOR_RGB2BGR))


    #       #       #       #       #       #       #       #
    # FILTER REFINEMENT     #       #       #

    @staticmethod
    def maskRefinement(mask,morph_ops=' ',k_size=5,k_shape=cv2.MORPH_ELLIPSE):
        """
        refine the mask w/ morphological operations

        :param mask: input mask to be refined
        :param morph_ops: ORDERED string of required morph. operations
                    IF int is passed (as string) selects a predefined set,
                    increasing intensities of refinement for tested ops combinations
                        '0','': off
                        '1': c  LOWER
                        '2': ce
                        '3': ceo
                        '4': ceh
                        '5': ceoh
                        '6': ceho   HIGHER
        :param k_size: kernel characteristic size
        :param k_shape: kernel shape:cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE}

        TODO: refine number of iterations based on image, samples,...
        """
        if morph_ops=='' or '0'in morph_ops: pass
        elif '1'in morph_ops: morph_ops='c'
        elif '2'in morph_ops: morph_ops='ce'
        elif '3'in morph_ops: morph_ops='ceo'
        elif '4'in morph_ops: morph_ops='ceh'
        elif '5'in morph_ops: morph_ops='ceoh'
        elif '6'in morph_ops: morph_ops='ceho'

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

    #       #       #       #       #       #       #       #
    # SAMPLES TOOLS

    def loadAllSamples(self,sample_lib=''):
        """
        automatical sample loading from a specific directoty

        TODO: saved images must be already temporally-averaged to remove more noise
        """
        if sample_lib=='':
            sample_lib=self.sample_source
        sample_lib=glob(sample_lib+"*.jpg")

        if sample_lib:
            for sample in sample_lib:
                new_sampl=cv2.imread(sample)
                try:
                    new_sample = cv2.cvtColor(new_sampl, cv2.COLOR_BGR2HSV)
                    _, _, sample_hist = computeHistogram(new_sample, 'hs', range_normalize=True)
                    sample_hist = sample_hist.astype('float32')
                    self.samples.append(sample_hist)
                except Exception as exc:
                    print('sample aquisition aborted ' + str(exc)[13])
            print("number of current samples: {}\n".format(len(self.samples)))
        else:
            self.addedAllSample=False
            print('No salmple found in selected folder')

    def addNewSample(self,save_sample=False,compute_avg=False):
        """
        manual sample selection from image from a static window
        test of correlation w/ while image not activated (prevents remotion of PARTICULARS)

        :param save_sample: toggles sample save to a file in sample directory
        :param compute_avg: computes the average color instead of adding the sample
        :param compute_avg: computes the average color instead of adding the sample

        TODO: since selection occours over a single frame of video, it's prone to noise:
         should select a TEMPORAL MEAN of that region (but rover must stand still)

        TODO: no tested methid permitted to have the program working while selecting:
         not putting this in a different callback (freezes), spawning another listener,...
         ALSO: multithreading does not really exist thanks to GIL
         BUT: nogil

        TODO: comparison between histograms of samples and whole image/dominant colors
            will be required for automatic selection of samples
         histogram comparison methods and relative tested data (static setup)
         hist_compare_dict = {
            "Correlation", (cv2.HISTCMP_CORREL, 0.2),  # or  0.4
            "Chi-Squared", (cv2.HISTCMP_CHISQR, 10000000),
            "Intersection", (cv2.HISTCMP_INTERSECT, 0.45),
            "Hellinger", (cv2.HISTCMP_BHATTACHARYYA, 0),
            "Kullback-Leibler", (cv2.HISTCMP_KL_DIV, 500)}
         _,_,image_hist=computeHistogram(image,'hs',range_normalize=True)
         if cv2.compareHist(image_hist,sample_hist,hist_compare_method)>hist_compare_thres or bypass_comp:

        TODO: temporally average samples by collecting @same ROI for some time;
            once ROI coords are selected, use the image stream for some time
            https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#addweighted
            https://leslietj.github.io/2020/06/28/How-to-Average-Images-Using-OpenCV/
            https://stackoverflow.com/questions/20175143/get-the-mean-of-sequence-of-frames
        """
        # IMGS_TO_ACCUMULATE=5
        # ACCUMULATOR_WEIGHT=1/IMGS_TO_ACCUMULATE
        winname="MOUSE+ENTER for selection,ESC to exit"
        img=self.preproc_img
        if not compute_avg:
            rois=cv2.selectROIs(winname,img)
            # rois=cv2.selectROIs(winname,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.destroyWindow(winname)
            for (roi_x0,roi_y0,roi_x,roi_y) in rois:
                try:
                    new_sampl=img[roi_y0:roi_y+roi_y0,roi_x0:roi_x+roi_x0]
                    new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_BGR2HSV)
                    # new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_RGB2HSV)
                    _,_,sample_hist=computeHistogram(new_sample,'hs',range_normalize=True)
                    sample_hist=sample_hist.astype('float32')

                    #TODO: sample accumulation (temporal mean) to reject noise
                    # requires an action, otherwise will always block waiting for this
                    # sample_accumulator=np.zeros((roi_y,roi_x,self.camera_channels))
                    # for count in range(IMGS_TO_ACCUMULATE):
                    #     img=self.preproc_img
                    #     cv2.imshow('img', img)
                        # new_sampl=img[roi_y0:roi_y+roi_y0,roi_x0:roi_x+roi_x0]
                        # new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_RGB2HSV)
                        # cv2.imshow('sample', new_sample)
                        # time.sleep(2)
                        # # cv2.accumulateWeighted(new_sample,sample_accumulator,ACCUMULATOR_WEIGHT)
                        # cv2.accumulate(new_sample,sample_accumulator)
                        # cv2.imshow('acc', sample_accumulator)
                    # sample_accumulator/=IMGS_TO_ACCUMULATE
                    # _,_,sample_hist=computeHistogram(sample_accumulator,'hs',range_normalize=True)
                    self.samples.append(sample_hist)
                except Exception as exc:
                    print('sample aquisition aborted '+str(exc)[13])
                if save_sample:
                    filename=self.sample_source+'sample_{}.jpg'.format(str(time.time()))
                    try:
                        cv2.imwrite(filename,new_sampl)
                        # cv2.imwrite(filename,cv2.cvtColor(new_sampl,cv2.COLOR_RGB2BGR))
                        print('saved: {}'.format(filename))
                    except Exception as exc:
                        print('sample saving unsuccessfull '+str(exc)[13])
            print("number of current samples: {}".format(len(self.samples)))
        else:
            roi=cv2.selectROI(winname,cv2.cvtColor(img,cv2.COLOR_BGR2BGR))
            # roi=cv2.selectROI(winname,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.destroyWindow(winname)
            roi_x0,roi_y0,roi_x,roi_y= roi
            new_sampl=img[roi_y0:roi_y+roi_y0,roi_x0:roi_x+roi_x0]
            new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_BGR2HSV)
            # new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_RGB2HSV)
            arr = np.reshape(new_sample, (new_sample.shape[0] * new_sample.shape[1], 3))
            print('HSV avg: {}'+format(np.mean(arr,axis=0)))

    def removeAllSamples(self):
        """
        removes all samples in self.sample array
        """
        self.samples=[]
        print("number of current samples: {}\n".format(len(self.samples)))


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   VISUALIZATION

    def showChannels(self,noisy='p',channels=' ',downsize_factor=1,winname_prefix='',h_shift=0):
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
        if noisy == 'p' or noisy == 'preprocessed':
            image = self.preproc_img
            winname_prefix = winname_prefix + ' PREPROCESSED '
        # elif noisy == 'n' or noisy == 'noisy':
        #     image = self.noisy_img
        #     winname_prefix = winname_prefix + ' NOISY '
        # else:
        #     image = self.sim_img
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
            hsv_in = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
            # hsv_in = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
            if 'h' in channels:
                out = (hsv_in[:, :, 0] + h_shift) % 180
                cv2.imshow(winname_prefix + "H", out)
            if 's' in channels:
                cv2.imshow(winname_prefix + "S", hsv_in[:, :, 1])
            if 'v' in channels:
                cv2.imshow(winname_prefix + "V", hsv_in[:, :, 2])
        if 'GL' in channels:
            gl_out = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            # gl_out = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
            cv2.imshow(winname_prefix + "GL", gl_out)


    def showHistograms(self,channels,noisy='p',winname_prefix=''):
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
        if noisy == 'p' or noisy == 'preprocessed':
            rgb_in_image = self.preproc_img
            rgb_in_image=cv2.cvtColor(rgb_in_image,cv2.COLOR_BGR2RGB)
            winname_prefix = winname_prefix + ' PREPROCESSSED '
        elif noisy == 'n' or noisy == 'noisy':
            winname_prefix = winname_prefix + ' NOISY '
            rgb_in_image = self.noisy_img
            rgb_in_image=cv2.cvtColor(rgb_in_image,cv2.COLOR_BGR2RGB)
        # else:
        #     rgb_in_image = self.sim_img
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
            # hsv_in_image = cv2.cvtColor(rgb_in_image.copy(), cv2.COLOR_BGR2HSV)
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
            # gl_out = cv2.cvtColor(rgb_in_image, cv2.COLOR_BGR2GRAY)
            gl_out = cv2.cvtColor(rgb_in_image, cv2.COLOR_RGB2GRAY)
            cv2.imshow(winname_prefix + "GL", gl_out)


    #   #   #   #   #   #   #   #   #   #   #   #
    # MENU
    def keyAction(self,key):
        """
        selects which action to perform pressing the key
        NOTE: must have the result window highlighted
        :param key:
            s: displays sample filter
            r: displays range filter
            o: displays otsu filter         --- REMOVED
            b: displays balancing filter    --- REMOVED
            z: enables all samples in ../media/samples folder for sample filter
            x: disables all samples
            l: add new sample(s) to filter from ROI(s) selection;
                multiple allowed, esc to exit
            k: same as 'k' & new samples are saved in ./media/samples folder
            m: increase morphological operations amount
            n: decrease morphological operations amount
            d: select depth based (sky) prefilter
            c: select color based (sky) prefilter

        TODO: there is to be, for sure, a more compact way to do so
        """
        if key ==ord('r'):
            cv2.destroyAllWindows()
            self.toggle_sample = False
            self.toggle_range = True
            self.resetStatistics()
        elif key == ord('s'):
            cv2.destroyAllWindows()
            self.toggle_range = False
            self.toggle_sample = True
            self.resetStatistics()
        elif key == ord('m'):
            # self.toggle_morph = False if self.toggle_morph else True
            try:
                morph_amount=int(self.MORPH_OPS)
                if morph_amount<6:
                    self.MORPH_OPS=str(morph_amount+1)
            except ValueError:#catches int(NON_NUMERIC_STR)
                self.MORPH_OPS ='0'
        elif key == ord('n'):
            try:
                morph_amount=int(self.MORPH_OPS)
                if morph_amount>0:
                    self.MORPH_OPS=str(morph_amount-1)
            except ValueError:
                self.MORPH_OPS ='0'
        elif key == ord('z') and not self.addedAllSample:
            self.addedAllSample=True
            cv2.destroyAllWindows()
            self.loadAllSamples()
            self.resetStatistics()
        elif key == ord('x') and len(self.samples)>0:
            self.addedAllSample=False
            cv2.destroyAllWindows()
            self.removeAllSamples()
            self.resetStatistics()
        elif key == ord('k'):
            self.addNewSample(save_sample=True)
            cv2.destroyAllWindows()
        elif key == ord('l'):
            self.addNewSample()
            cv2.destroyAllWindows()
        elif key == ord('d'):
            self.selected_premask='d'
        elif key == ord('c'):
            self.selected_premask='c'
        #TODO
        elif key == ord('i'):
            self.addNewSample(compute_avg=True)


    #   #   #   #   #   #   #   #   #   #   #   #
    # STATISTICS
    def resetStatistics(self):
        self.ave_time=0
        self.iter_counter=0


    def updateStatistics(self,time_setpoint):
        """
        running average for cycle time
        pause required for misleading results when awiting for user input
        """
        self.iter_stamp+=1
        if not self.pause_stats:
            self.ave_time = (self.ave_time * self.iter_counter + time.time() - time_setpoint) / (self.iter_counter + 1)
            print(' avg. cycle [ms]: {}'.format(np.round(self.ave_time * 1000, 6)), end='\r')
            self.iter_counter += 1
        else:
            self.pause_stats=False
        self.this_time=time.time()


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
if __name__ == '__main__':
    node_name = 'image_filter'
    rospy.init_node(node_name, anonymous=False)
    print("\nNavCam Image Prepocessing Node for Navigation Pipeline")
    print('node name: ' + node_name)

    preprocessor = GroundFilter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit(0)
    rospy.loginfo("exiting...")

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

DISTANZA DI COLORE (MAHALANOBIS, ...)

    
    https://github.com/tvanslyke/ImageProcessing/blob/master/Mahalanobis.py
    https://dsp.stackexchange.com/questions/1625/basic-hsb-skin-detection-neon-illumination


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

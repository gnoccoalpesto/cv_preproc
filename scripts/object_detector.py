#!/usr/bin/env python3
"""
AUTHOR:     FRANCESCO CERRI
            francesco.cerri2@studio.unibo.it

            ALMA-X TEAM
            UNIVERSITÀ DEGLI STUDI DI BOLOGNA ALMA MATER STUDIORUM

THIS CODE DETECTS OBJECTS IN THE VIDEO STREAM PUBLICHED BY
IMAGE FILTER NOODE

RELATED WORKS (kinda messy, requires iphyton capable sys):
    https://github.com/alma-x/CV-Alma-X/

--------
WHAT THIS CLASS DOES:
0) establish connection with ROS master node
0.1) waits for incoming filtered camera messages

1) edge detection

2) closed contours detection

3) objects analysis based on some statistics

-----------------------
IDEAS
    visual markers have a box shape

    the majority of bigger rocks will have an ellypsoid shape

    remaining terrain patches will have mostly an irregular shape

    also, color thresholding could be still a useful thing

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
lib_histograms              included

SIMULATION (provides all the ingoing data)
https://github.com/EuropeanRoverChallenge/ERC-Remote-Navigation-Sim

NOTE: packages denoted by REQUIRED* are intended for ROS usage
        if not present, use the noROS_NAME.py version (loads a saved video)
"""
# python ROS implementation
import rospy
# ROS message types: required for I/O
from sensor_msgs.msg import Image  # ,Imu
from std_msgs.msg import String, Bool
# ROS image <--> cv/np image conversion
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
# histograms analysis tools and filters definitions
from lib_histograms import computeHistogram, drawHistogram
from lib_filters import bilateralFilter,sobelFilter,highPassFilter,\
    linearStretching,findPercentileValue,equalization,pfm
# TODO: fix import
# TODO? ALSO import class methods?
import time
from glob import glob


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
class ObjectDetectorNode:
    """    """

    def __init__(self):

        self.input_topic = rospy.get_param(
            '/image_preproc/filtered_camera_topic','/camera/image_raw/filtered')

        print('listened topic: ' + self.input_topic)

        self.filtered_img = np.ndarray
        self.filtered_grey= np.ndarray
        self.in_dtype = None
        self.camera_channels = int
        self.current_resolution = tuple

        self.cvbridge = CvBridge()

        # TODO: add rosparams to change stuff at runtime

        # depth data already subsampled of same amount of image
        self.depth_topic = '/zed2/depth/depth_preproc'

        # obtains a ~10ms cycle len, considering a bit of superimposed overhead
        DETECTOR_BASE_FREQ = 100
        detector_freq = rospy.get_param('image_preproc/detector_freq', DETECTOR_BASE_FREQ)
        detector_cycle_time_nsec = rospy.Duration(nsecs=int(1E9 / detector_freq))

        # self.MORPH_OPS = '1'  # =='c'
        # self.toggle_morph = True

        self.objects = []
        self.edges_objects = []
        self.contours_objects = []
        self.depth_data = np.ndarray

        #   PERFORMANCES   #   #   #   #   #   #
        self.pause_stats = False
        self.iter_counter = 0
        self.ave_time = 0

        #   KCLUSTERS     #   #   #   #   #   #
        self.n_colors = 12
        self.cl_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.cl_flags = cv2.KMEANS_RANDOM_CENTERS

        #   #   #   #   #   #   #   #   #
        self.initDetector()

        #   ROS I/O OPERATIONS   #   #   ##   #   #   #   #   ##   #   #   #   #   #
        self.inputListener = rospy.Subscriber(self.input_topic, Image, self.inputCallback)
        # self.morphOpsListener = rospy.Subscriber(self.morph_ops_source, String, self.morphOpsCallback)
        self.depthListener = rospy.Subscriber(self.depth_topic, Image, self.depthCallback)
        time.sleep(.25)
        self.this_time = time.time()
        # overhead can be controlled by setting a different cycle frequency
        self.detectorTimer = rospy.Timer(detector_cycle_time_nsec, self.detectorCallback)


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   INITIALIZATION, I/O CALLBACKS, STATISTICS   #   #   #   #   #   #
    def initDetector(self):
        print('initializing...')
        img_msg = rospy.wait_for_message(self.input_topic, Image)
        image = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        self.filtered_img = image
        self.in_dtype = image.dtype
        self.camera_channels = image.shape[2]
        self.current_resolution = image.shape[:2]
        print("INPUT\t==\t==\t==\n resolution: {}\n channels: {}\n depth type: {}".
              format(str(self.current_resolution), str(self.camera_channels), str(self.in_dtype)))


    # def morphOpsCallback(self, str_msg):
    #     # TODO version w/ timered/every-cycle rosparam reading (get) overhead?
    #     """
    #     whenerever a morph_ops string is received, updates self.MORPH_OPS
    #     """
    #     self.MORPH_OPS = str_msg.data


    def depthCallback(self, depth_msg):
        """
        acquires depth camera information
        then computes a mask based on a certaint threshold (nan: threshold=max sensor range)

        TODO: modify to implement variable distance thresholding; could be usefull
            to focus attention on close areas;
                .astype('float32') gives values decreasing with proximity (min ~0.4)
                BUT still nan outside usefull range
        """
        try:
            self.depth_data = self.cvbridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
        except CvBridgeError:
            print("DEPTH: cv bridge error")


    def inputCallback(self, img_msg):
        """
        'Real Time' (threaded) ascquisition of preprocessed input image
        """
        try:
            image = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            self.filtered_img=image
            self.filtered_grey=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except CvBridgeError:
            print("INPUT: cv bridge error")


    #    ##################################################################################
    ###    ##################################################################################
    ######    ##################################################################################

    # MAIN LOOP : every time a msg on self.input_topic received
    def detectorCallback(self, _):
        self.updateStatistics(self.this_time)
        # morph_ops = self.MORPH_OPS  # if self.toggle_morph else ''

        image=self.filtered_img.copy()
        grey_image=self.filtered_grey.copy()

        # image_hist=computeHistogram(image,'r',range_normalize=True)

        # PREPROCESSING --------------------------------------------------------------
        # self.cameraClustering()


        # BLOBS ----------------------------------------------------------------------
        # this causes problems with marker's black part;
        # YET they are always in the interior; HENCE could solve by FILLING
        blob_img=np.zeros_like(self.filtered_grey.copy())
        blob_img[grey_image>0]=255
        blob_img=self.objectRefinement(blob_img,'c')
        cv2.imshow('obj morph',blob_img)

        sobel_img=sobelFilter(image.copy())
        cv2.imshow('sobel img',sobel_img)
        sobel_grey=sobelFilter(grey_image.copy())
        cv2.imshow('sobel grey',sobel_grey)

        # stretched_img=linearStretching(image.copy(),'p5','p95',image_hist)
        # cv2.imshow('stretched img',stretched_img)

        # improving contrast, may help localizing edges
        # equalized_img=equalization(image.copy(),image_hist)
        # cv2.imshow('equalized img',equalized_img)


        # EDGES -------------------------------------------------------------------------
        high_passed_img=highPassFilter(image.copy())
        cv2.imshow('high pass img',high_passed_img)
        high_passed_grey=highPassFilter(image.copy())
        cv2.imshow('high pass grey',high_passed_grey)

        filterparam = (100, 20, 0)
        grey_edges = cv2.Canny(grey_image, filterparam[0], filterparam[1],filterparam[2])
        cv2.imshow("grey canny's",grey_edges)
        blob_edges = cv2.Canny(blob_img, filterparam[0], filterparam[1],filterparam[2])
        cv2.imshow("blob canny's",blob_edges)
        sobel_edges = cv2.Canny(sobel_img, filterparam[0], filterparam[1],filterparam[2])
        cv2.imshow("sobel canny's",sobel_edges)
        sobel_grey_edges = cv2.Canny(sobel_grey, filterparam[0], filterparam[1],filterparam[2])
        cv2.imshow("sobel blob canny's",sobel_grey_edges)

        # computes external edges
        # equalized_edges = cv2.Canny(equalized_img.astype('uint8'), filterparam[0], filterparam[1],filterparam[2])
        # cv2.imshow("canny's equalized",equalized_edges)


        # CLEANING -------------------------------------------------------------------------



        
        # CLOSED CONTOURS -------------------------------------------------------------------------

        ##w/ canny
        # cnts_can = cv2.findContours(out_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        ##w/out canny
        # cnts_nocan = cv2.findContours(self.filtered_grey, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]


        # selected_cnts=cnts_nocan
        # selected_out=[]
        # cnt_img= self.filtered_grey.copy()


        def testShape(contours,n_edges,test_method='equals'):
            """
            test contours based on number of edges
            :param contours: array of candidate contours
            :param n_edges: number of edges to look for
            :param test_method: 'equals': exatly that many edges
                                'atleat': at least that many edges
                                'atmost' UNUSED
            :return: tested contours array (reduced input)
            """
            tested_contours=[]
            if n_edges==0: return contours
            perimeter_coeff = .1
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx_contour=cv2.approxPolyDP(contour, perimeter_coeff * perimeter, True)
                # ar = w / float(h)
                if (test_method=='equals' and len(approx_contour)==n_edges) \
                 or (test_method=='atleast' and len(approx_contour)>=n_edges):
                    tested_contours.append(contour)
            return tested_contours

        def testArea(contours,test_method='maxa',min_area=None,max_area=None):
            """
            test contours based on number of edges
            :param contours: array of candidate contours
            :param test_method: 'min': bigger than min_area
                                'max*': percentage of max_area
                                'maxl': uses area of longest contour
                                'maxa' uses area of max area contour
            :param min_area: UNUSED
            :param max_area: if None, selects area among contours
            :return: tested contours array (reduced input)

            NOTE: can also use approx_contour area
            """
            tested_contours=[]
            area_coeff = .1
            if max_area is None:
                if 'l'in test_method:
                    max_lenght_contour=max(contours,key=len)
                    max_area = cv2.contourArea(max_lenght_contour)
                else:
                    max_area = max(contours, key=cv2.contourArea)
            for contour in contours:
                area_contour=cv2.contourArea(contour)
                if test_method=='max' and area_contour >= area_coeff * max_area:
                    tested_contours.append(contour)
            return tested_contours



        ##opening/closing for removing noise: better to be done after contours found
        # contours selection
        ##improve:dilate before comparison and pick the inner areas

        ##cherrypiching with hough? moments?
        ## alsoM = cv2.moments(c)
        ## center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # draw contours
        # cnt_img=np.zeros_like(self.filtered_img)
        # lots of smaller contours
        # for c in selected_cnts:
        #     cv2.drawContours(cnt_img, c, -1, (255, 255, 0), 5)  # OR
            # cv2.polylines(image, [box], True, (36,255,12), 3)
        # cv2.drawContours(cnt_img, selected_cnts, -1, contcolor, conthick)
        # area contour works great with solid shapes
        # cv2.drawContours(cnt_img,[max(selected_cnts, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        # long contours works great with canny's
        # cv2.drawContours(cnt_img,[max(selected_cnts, key=len)], -1, (0, 0, 255), 2)
        # cv2.imshow('contours',cnt_img)

        ########
        # CHANNEL-WISE CONTOURS
        # for channel in range(img.shape[2]):
        #     ret, image_thresh = cv.threshold(img[:, :, channel], 38, 255, cv.THRESH_BINARY)
        #     cnts = cv.findContours(image_thresh, 1, 1)[0]




        # OBJECTS -------------------------------------------------------------------------
        ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc
            rospy.signal_shutdown('Esc key pressed')
        elif k != 255:  # NO KEY PRESSED
            # print(k)
            self.keyAction(k)

    ######    ##################################################################################
    ###    ##################################################################################
    #    ##################################################################################

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   CONTOURS

    # def watershed(self):
    #     # laplacian filter
    #     kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    #     imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
    #     imgResult = np.float32(src) - imgLaplacian
    #     # convert back to 8bits gray scale
    #     imgResult = np.clip(imgResult, 0, 255)
    #     imgResult = imgResult.astype('uint8')
    #     imgLaplacian = np.clip(imgLaplacian, 0, 255)
    #     imgLaplacian = np.uint8(imgLaplacian)
    #     if debug:
    #         plt.imshow(imgResult, cmap='gray')
    #         plt.show()
    #     bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    #     _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     if debug:
    #         plt.imshow(bw, cmap='gray')
    #         plt.show()
    #     dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    #     # Normalize the distance image for range = {0.0, 1.0}
    #     # so we can visualize and threshold it
    #     cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    #     if debug:
    #         plt.imshow(dist, cmap='gray')
    #         plt.show()
    #     _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    #
    #     # Dilate a bit the dist image
    #     kernel1 = np.ones((3, 3), dtype=np.uint8)
    #     dist = cv2.dilate(dist, kernel1)
    #     if debug:
    #         plt.imshow(dist, cmap='gray')
    #         plt.show()
    #     dist_8u = dist.astype('uint8')
    #
    #     # Find total markers
    #     contours = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = contours[0] if len(contours) == 2 else contours[1]
    #
    #     # Create the marker image for the watershed algorithm
    #     markers = np.zeros(dist.shape, dtype=np.int32)
    #
    #     # Draw the foreground markers
    #     for i in range(len(contours)):
    #         cv2.drawContours(markers, contours, i, (i + 1), -1)  ##not working
    #     # cv2.drawContours(markers, contours, -1,(255,0,0), -1)
    #     # Draw the background marker
    #     cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    #     if debug:
    #         plt.imshow(markers, cmap='gray')
    #         plt.show()
    #
    #     cv2.watershed(imgResult, markers)
    #     # mark = np.zeros(markers.shape, dtype=np.uint8)
    #     mark = markers.astype('uint8')
    #     mark = cv2.bitwise_not(mark)
    #     if debug:
    #         plt.imshow(mark, cmap='gray')
    #         plt.show()
    #
    #     # Generate random colors
    #     colors = []
    #     for contour in contours:
    #         colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    #     # Create the result image
    #     dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    #
    #     # Fill labeled objects with random colors
    #     for i in range(markers.shape[0]):
    #         for j in range(markers.shape[1]):
    #             index = markers[i, j]
    #             if index > 0 and index <= len(contours):
    #                 dst[i, j, :] = colors[index - 1]

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   IMAGE PREPROCESSING
    def cameraClustering(self, cl_method='sy', show_result=False):
        """
        clustering routine using k-means
        :param cl_method: used method for clustering
        :param show_result: shows result in a window
        Tested:
            'cv2' or 'cv' or 'c'
            'scipy' or 'sy' or 'y'      (compatible speed, ~80ms overhead)
            'sklearn' or 'sk' or 'k'    (awkwardly slow)

        TODO initialize this offline to find dominant colors of image

        TODO: mask black pixels to avoid considering them in clustering
        """
        image = self.preproc_img
        current_resolution = self.current_resolution
        image = image.reshape((-1, 3)).astype('float32')
        if 'c' in cl_method:
            _, labels, centers = cv2.kmeans(image, self.n_colors, None, self.cl_criteria, 10, self.cl_flags)
        elif 'y' in cl_method:
            centers, labels = kmeans2(image, self.n_colors, 10, 1, 'points', 'warn', False)
        centers = np.uint8(centers)
        clustered = centers[labels.flatten()]
        h, w = current_resolution
        clustered = clustered.reshape((h, w, self.camera_channels))
        # TODO: dominant colors
        # _, counts = np.unique(labels, return_counts=True)
        # dominant = centers[np.argmax(counts)]
        # label_counts = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        # histogram, _ = np.histogram(cluster.labels_, bins=label_counts)
        # histogrdekam = histogram.astype('float32')
        # histogram /= histogram.sum()
        if show_result:
            cv2.imshow('clustered', cv2.cvtColor(clustered, cv2.COLOR_RGB2BGR))


    #       #       #       #       #       #       #       #
    # REFINEMENT     #       #       #

    @staticmethod
    def objectRefinement(mask, morph_ops=' ', k_size=5, k_shape=cv2.MORPH_ELLIPSE):
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
        if morph_ops == '' or '0' in morph_ops:
            pass
        elif '1' in morph_ops:
            morph_ops = 'c'
        elif '2' in morph_ops:
            morph_ops = 'ce'
        elif '3' in morph_ops:
            morph_ops = 'ceo'
        elif '4' in morph_ops:
            morph_ops = 'ceh'
        elif '5' in morph_ops:
            morph_ops = 'ceoh'
        elif '6' in morph_ops:
            morph_ops = 'ceho'

        kernel = cv2.getStructuringElement(k_shape, (k_size, k_size))
        for mop in morph_ops:
            if mop == 'd' or mop == 'dilate':
                mask = cv2.dilate(mask.copy(), kernel, iterations=1)
                # NOTE: mask=cv2.morphologyEx(mask.copy(), cv2.MORPH_DILATE, kernel)
            elif mop == 'e' or mop == 'erode':
                mask = cv2.erode(mask.copy(), kernel, iterations=1)
            elif mop == 'h' or mop == 'hitmiss':
                mask = cv2.morphologyEx(mask.copy(), cv2.MORPH_HITMISS, kernel)
            elif mop == 'o' or mop == 'open':
                mask = cv2.morphologyEx(mask.copy(), cv2.MORPH_OPEN, kernel)
            elif mop == 'c' or mop == 'close':
                mask = cv2.morphologyEx(mask.copy(), cv2.MORPH_CLOSE, kernel)
        return mask


    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    #   VISUALIZATION

    # def showChannels(self, noisy='p', channels=' ', downsize_factor=1, winname_prefix='', h_shift=0):
    #     """
    #     Shows selected image channels
    #     :param noisy: toggles sim or noisy input
    #     :param channels: permitted 'hsvrgbGL' (cumulative string)
    #             h s v : HSV colorspace
    #             r g b : RGB colorspace
    #             GL    : GreyLevels
    #             f     : full rgb image
    #     :param downsize_factor: final dimensions will be devided by this value
    #         NOTE: intended to be a "cosmetics" tool for lower screen resolutions
    #     :param winname_prefix: window text prefix
    #     :param h_shift: shift the h channel value values toward rigth of the histogram
    #                     highlighting similarity of values around 0==180 (especially for noisy imgs)
    #     """
    #     if noisy == 'p' or noisy == 'preprocessed':
    #         image = self.preproc_img
    #         winname_prefix = winname_prefix + ' PREPROCESSED '
    #     # elif noisy == 'n' or noisy == 'noisy':
    #     #     image = self.noisy_img
    #     #     winname_prefix = winname_prefix + ' NOISY '
    #     # else:
    #     #     image = self.sim_img
    #     if downsize_factor != 1:
    #         new_size = (int(image.shape[1] / downsize_factor), int(image.shape[0] / downsize_factor))
    #         image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    #     if 'f' in channels:
    #         rgb_out = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    #         cv2.imshow(winname_prefix + 'FULL RGB ', rgb_out)
    #     if 'r' in channels:
    #         cv2.imshow(winname_prefix + "R", image[:, :, 0])
    #     if 'g' in channels:
    #         cv2.imshow(winname_prefix + "G", image[:, :, 1])
    #     if 'b' in channels:
    #         cv2.imshow(winname_prefix + "B", image[:, :, 2])
    #     if 'h' in channels or 's' in channels or 'v' in channels:
    #         hsv_in = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
    #         if 'h' in channels:
    #             out = (hsv_in[:, :, 0] + h_shift) % 180
    #             cv2.imshow(winname_prefix + "H", out)
    #         if 's' in channels:
    #             cv2.imshow(winname_prefix + "S", hsv_in[:, :, 1])
    #         if 'v' in channels:
    #             cv2.imshow(winname_prefix + "V", hsv_in[:, :, 2])
    #     if 'GL' in channels:
    #         gl_out = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    #         cv2.imshow(winname_prefix + "GL", gl_out)
    #
    # def showHistograms(self, channels, noisy='p', winname_prefix=''):
    #     """
    #     computes and shows H,S and RGB histograms
    #     :param noisy: toggles sim or noisy input
    #     :param channels: which channel to analyze and show
    #     :param winname_prefix: window text prefix
    #     :internal:  xyz_scaling: img height rescaling,
    #                 zyx_shift: shift toward right side of hist (higher values)
    #
    #     TODO: optional superimposing of noisy histogram
    #     """
    #     rgb_scaling = 0.01
    #     hsv_shift = 15
    #     rgb_shift = 92
    #     if noisy == 'p' or noisy == 'preprocessed':
    #         rgb_in_image = self.preproc_img
    #         winname_prefix = winname_prefix + ' PREPROCESSSED '
    #     # elif noisy == 'n' or noisy == 'noisy':
    #     #     winname_prefix = winname_prefix + ' NOISY '
    #     #     rgb_in_image = self.noisy_img
    #     # else:
    #     #     rgb_in_image = self.sim_img
    #     shift_text = ", SHIFT AMOUNT= " + str(int(rgb_shift * 2.56)) + " /256"
    #     if 'r' in channels or 'f' in channels:
    #         r_hist, _ = computeHistogram(rgb_in_image, 'r')
    #         r_hist_img = drawHistogram(r_hist, img_zoom=3, shift_percent=rgb_shift, man_scale=rgb_scaling)
    #         if 'f' not in channels:
    #             cv2.imshow(winname_prefix + "R" + shift_text, r_hist_img)
    #     if 'g' in channels or 'f' in channels:
    #         g_hist, _ = computeHistogram(rgb_in_image, 'g')
    #         g_hist_img = drawHistogram(g_hist, img_zoom=3, shift_percent=rgb_shift, man_scale=rgb_scaling)
    #         if 'f' not in channels:
    #             cv2.imshow(winname_prefix + "G", g_hist_img)
    #     if 'b' in channels or 'f' in channels:
    #         b_hist, _ = computeHistogram(rgb_in_image, 'b')
    #         b_hist_img = drawHistogram(b_hist, img_zoom=3, shift_percent=rgb_shift, man_scale=rgb_scaling)
    #         if 'f' not in channels:
    #             cv2.imshow(winname_prefix + "B", b_hist_img)
    #     if 'f' in channels:
    #         rgb_hist_img = cv2.merge([b_hist_img, g_hist_img, r_hist_img])
    #         cv2.imshow(winname_prefix + "RGB HIST" + shift_text, rgb_hist_img)
    #
    #     if 'h' in channels or 's' in channels or 'v' in channels:
    #         shift_text = ", SHIFT AMOUNT= " + str(int(hsv_shift * 1.8)) + " /180"
    #         hsv_in_image = cv2.cvtColor(rgb_in_image.copy(), cv2.COLOR_RGB2HSV)
    #         if 'h' in channels:
    #             _, _, h_hist = computeHistogram(hsv_in_image, 'h', range_normalize=True)
    #             h_hist_img = drawHistogram(h_hist, img_zoom=3, shift_percent=hsv_shift)
    #             h_hist_img = h_hist_img[h_hist_img.shape[0] // 2:, :]
    #             cv2.imshow(winname_prefix + "H" + shift_text, h_hist_img)
    #         if 's' in channels:
    #             _, _, s_hist = computeHistogram(hsv_in_image, 's', range_normalize=True)
    #             s_hist_img = drawHistogram(s_hist, img_zoom=3)
    #             s_hist_img = s_hist_img[s_hist_img.shape[0] // 2:, :]
    #             cv2.imshow(winname_prefix + "S" + " NOT SHIFTED", s_hist_img)
    #         if 'v' in channels:
    #             _, _, v_hist = computeHistogram(hsv_in_image, 'v', range_normalize=True)
    #             v_hist_img = drawHistogram(v_hist, img_zoom=3)
    #             v_hist_img = v_hist_img[v_hist_img.shape[0] // 2:, :]
    #             cv2.imshow(winname_prefix + "V" + " NOT SHIFTED", v_hist_img)
    #     if 'GL' in channels:
    #         gl_out = cv2.cvtColor(rgb_in_image, cv2.COLOR_RGB2GRAY)
    #         cv2.imshow(winname_prefix + "GL", gl_out)

    #   #   #   #   #   #   #   #   #   #   #   #
    # MENU
    # def keyAction(self, key):
    #     """
    #     :param key:
    #         s: displays sample filter
    #         r: displays range filter
    #         o: displays otsu filter
    #         b: displays balancing filter
    #         z: enables all samples in ../media/samples folder for sample filter
    #         c: disables all samples
    #         l: add new sample(s) to filter from ROI(s) selection;
    #             multiple allowed, esc to exit
    #         k: same as 'k' & new samples are saved in ./media/samples folder
    #         m: toggles morphological operations
    #
    #     TODO: there is to be, for sure, a more compact way to do so
    #     """
    #     if key == ord('r'):
    #         cv2.destroyAllWindows()
    #         self.toggle_sample = False
    #         self.toggle_range = True
    #         self.resetStatistics()
    #     elif key == ord('s'):
    #         cv2.destroyAllWindows()
    #         self.toggle_range = False
    #         self.toggle_sample = True
    #         self.resetStatistics()
    #     elif key == ord('m'):
    #         # self.toggle_morph = False if self.toggle_morph else True
    #         try:
    #             morph_amount = int(self.MORPH_OPS)
    #             if morph_amount < 6:
    #                 self.MORPH_OPS = str(morph_amount + 1)
    #         except ValueError:  # catches int(NON_NUMERIC_STR)
    #             self.MORPH_OPS = '0'
    #     elif key == ord('n'):
    #         try:
    #             morph_amount = int(self.MORPH_OPS)
    #             if morph_amount > 0:
    #                 self.MORPH_OPS = str(morph_amount - 1)
    #         except ValueError:
    #             self.MORPH_OPS = '0'
    #     elif key == ord('z') and not self.addedAllSample:
    #         self.addedAllSample = True
    #         cv2.destroyAllWindows()
    #         self.loadAllSamples()
    #         self.resetStatistics()
    #     elif key == ord('c') and len(self.samples) > 0:


    #   #   #   #   #   #   #   #   #   #   #   #
    # STATISTICS
    def resetStatistics(self):
        self.ave_time = 0
        self.iter_counter = 0

    def updateStatistics(self, time_setpoint):
        """
        running average for cycle time
        pause required for misleading results when awiting for user input
        """
        if not self.pause_stats:
            self.ave_time = (self.ave_time * self.iter_counter + time.time() - time_setpoint) / (self.iter_counter + 1)
            print(' avg. cycle [ms]: {}'.format(np.round(self.ave_time * 1000, 6)), end='\r')
            self.iter_counter += 1
        else:
            self.pause_stats = False
        self.this_time = time.time()


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
if __name__ == '__main__':
    node_name = 'object_detector'
    rospy.init_node(node_name, anonymous=False)
    print("\nNavCam Object Detector for Navigation Pipeline")
    print('node name: ' + node_name)

    preprocessor = ObjectDetectorNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit(0)
    rospy.loginfo("exiting...")

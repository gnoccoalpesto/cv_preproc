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
# filters definitions
from lib_filters import bilateralFilter,sobelFilter,highPassFilter,gaussianFilter, medianFilter
import time


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
class ObjectDetector:
    """    """

    def __init__(self):

        self.input_topic = rospy.get_param(
            '/image_preproc/filtered_camera_topic','/camera/image_raw/filtered')

        print('listened topic: ' + self.input_topic)

        self.original_topic="/camera/image_raw/preproc"

        self.original_image=np.ndarray
        self.filtered_img = np.ndarray
        self.filtered_grey= np.ndarray
        self.in_dtype = None
        self.camera_channels = int
        self.current_resolution = tuple

        self.cvbridge = CvBridge()

        # TODO: add rosparams to change stuff at runtime

        # depth data already subsampled of same amount of image
        self.depth_topic = '/zed2/depth/depth_preproc'

        self.objects_topic='/cv_preproc/objects'

        # obtains a ~10ms cycle len, considering a bit of superimposed overhead
        DETECTOR_BASE_FREQ = 1000
        detector_freq = rospy.get_param('image_preproc/detector_freq', DETECTOR_BASE_FREQ)
        detector_cycle_time_nsec = rospy.Duration(nsecs=int(1E9 / detector_freq))

        self.toggle_prerefinement=True

        self.toggle_enclosing=True
        self.toggle_inner = True
        self.toggle_canny = False
        self.toggle_hipass = False
        self.objects = [] # UNUSED
        self.edges = np.ndarray
        self.contours = []
        self.depth_data = np.ndarray

        #   PERFORMANCES   #   #   #   #   #   #
        self.pause_stats = False
        self.iter_counter = 0
        self.ave_time = 0

        #   #   #   #   #   #   #   #   #
        self.initDetector()
        self.iter_count=0
        self.hipass_ave=0
        self.canny_ave=0

        #   ROS I/O OPERATIONS   #   #   ##   #   #   #   #   ##   #   #   #   #   #
        self.inputListener = rospy.Subscriber(self.input_topic, Image, self.inputCallback)
        # self.morphOpsListener = rospy.Subscriber(self.morph_ops_source, String, self.morphOpsCallback)
        self.depthListener = rospy.Subscriber(self.depth_topic, Image, self.depthCallback)
        self.original_sub = rospy.Subscriber(self.original_topic, Image, self.originalCallback)
        self.objects_pub=rospy.Publisher(self.objects_topic,Image,queue_size=1)
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
            self.filtered_grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # self.filtered_grey=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except CvBridgeError:
            print("INPUT: cv bridge error")


    def originalCallback(self, img_msg):
        """
        'Real Time' (threaded) ascquisition of preprocessed input image
        """
        try:
            image = self.cvbridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            self.original_image=image
        except CvBridgeError:
            print("INPUT: cv bridge error")
    #    ##################################################################################
    ###    ##################################################################################
    ######    ##################################################################################

    # MAIN LOOP : every time a msg on self.input_topic received
    def detectorCallback(self, _):
        # self.updateStatistics(self.this_time)
        # PREPROCESSING --------------------------------------------------------------
        grey_image=self.filtered_grey.copy()

        # BLOBS ----------------------------------------------------------------------
        if self.toggle_inner:
            window_name = "inner"
            blob_img = np.zeros_like(self.filtered_grey.copy())
            blob_img[grey_image > 0] = 255
            if self.toggle_prerefinement:
                blob_img = medianFilter(blob_img.copy(), 3)
                # almost no effect compared to median filter
                # blob_img=self.objectRefinement(blob_img,'o',k_size=3)
                # blob_img=self.objectRefinement(blob_img,'c',k_size=5)
                # self.PRE="ENABLED"
            # else: self.PRE="DISABLED"
            # self.DETECTOR="INNER, MORPH BASED"
        # EDGES -------------------------------------------------------------------------
            STRUCTURING_ELEMENT = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))
            self.edges = (cv2.dilate(blob_img.copy(), STRUCTURING_ELEMENT)) - blob_img

        elif self.toggle_canny:
            window_name="canny's"
            if self.toggle_prerefinement:
                grey_image=medianFilter(grey_image.copy(),3)
                # almost no effect compared to median filter
                # blob_img=self.objectRefinement(blob_img,'o',k_size=3)
                # blob_img=self.objectRefinement(blob_img,'c',k_size=5)

        # no real difference with changing params
            canny_thr_high=100
            canny_thr_low=20
            k_size_sobel=5
            use_L2_gradient=True
            self.edges = cv2.Canny(grey_image, canny_thr_high, canny_thr_low,
                                   apertureSize=k_size_sobel,L2gradient=use_L2_gradient)

        elif self.toggle_hipass:
            window_name="high pass filter"
            # reliabily spots the inner square of the marker
            # BUT lots of internal edges at low intensity
            hipass_edges=highPassFilter(grey_image)
            hipass_edges[hipass_edges>0]=255
            self.edges=hipass_edges

        # CLEANING -------------------------------------------------------------------------
        # no real use, best if done before of edges detection
        #TODO: could be smoothing->thresholding to remove "too smoothed" valuer (near "noise") be usefull to
        #   reject the smallest blobs/edges and noise?
        
        # CLOSED CONTOURS -------------------------------------------------------------------------
        self.contours = cv2.findContours(self.edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        # area contour works great with solid shapes
        # max_area_contour=max(self.contours, key=cv2.contourArea)
        # long contours works great with canny's
        # max_len_contour=max(self.contours, key=len)

        ##opening/closing for removing noise: better to be done after contours found
        # contours selection
        ##improve:dilate before comparison and pick the inner areas
        #TODO: draw contours on black image, then morph ops, then find edges and contours

        # CONTOURS SELECTION
        ## alsoM = cv2.moments(c)
        ## center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        #TODO: area filtering
        # self.contours=self.testArea(self.contours,'maxa',area_coeff=.1)

        self.contours=sorted(self.contours,key=lambda ctr: cv2.contourArea(ctr),reverse=True)
        big_contours=self.contours[:20]
        rectangular_contours=self.testShape(self.contours,4)
        rectangular_contours=sorted(rectangular_contours,key=lambda ctr: cv2.contourArea(ctr),reverse=True)
        big_rectangular_contours=rectangular_contours[:10]


        # OBJECTS -------------------------------------------------------------------------
        BGR_VIOLET=(190, 20, 100)
        RGB_PURPLE=(255,0,255)
        RGB_CYAN=(0,255,255)
        RGB_YELLOW=(255,255,0)
        RGB_VIOLET=(100,36,210)
        RGB_GREEN=(0,255,0)
        # cnt_img=np.zeros_like(self.filtered_img)
        cnt_img=cv2.cvtColor(self.original_image,cv2.COLOR_BGR2RGB)
        # cnt_img=self.original_image
        cv2.drawContours(cnt_img, self.contours, -1, RGB_VIOLET, 1)
        cv2.drawContours(cnt_img,rectangular_contours, -1, RGB_YELLOW, 1)
        if self.toggle_enclosing:
            for contour in big_rectangular_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cnt_img, (x, y), (x + w, y + h), RGB_GREEN, 2)
            for contour in big_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cnt_img, (x, y), (x + w, y + h), RGB_CYAN, 2)
        cnt_img=cv2.cvtColor(cnt_img,cv2.COLOR_RGB2BGR)
        cv2.imshow('contours',cnt_img)

        obj_msg=self.cvbridge.cv2_to_imgmsg(cnt_img)
        self.objects_pub.publish(obj_msg)

        # TODO: cv2.pollKey() better because saves few msecs?
        # k = cv2.waitKey(1) & 0xFF
        k = cv2.pollKey() & 0xFF
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

    @staticmethod
    def testShape(contours, n_edges, test_method='equals'):
        """
        test contours based on number of edges
        :param contours: array of candidate contours
        :param n_edges: number of edges to look for
        :param test_method: 'equals': exatly that many edges
                            'atleat': at least that many edges
                            'atmost' UNUSED
        :return: tested contours array (reduced input)
        """
        tested_contours = []
        if n_edges == 0: return contours
        perimeter_coeff = .1
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, perimeter_coeff * perimeter, True)
            # ar = w / float(h)
            if (test_method == 'equals' and len(approx_contour) == n_edges) \
                    or (test_method == 'atleast' and len(approx_contour) >= n_edges):
                tested_contours.append(contour)
        return tested_contours


    @staticmethod
    def testArea(contours, test_method='maxa', min_area=None, max_area=None,area_coeff=.1):
        """
        test contours based on number of edges
        :param contours: array of candidate contours
        :param test_method: 'min': function of min_area
                            'max*': percentage of max_area
                            'maxl': uses area of longest contour
                            'maxa' uses area of max area contour
        :param min_area: UNUSED
        :param max_area: if None, selects area among contours
        :return: tested contours array (reduced input)

        NOTE: can also use approx_contour area
        """
        tested_contours = []
        if max_area is None:
            if 'l' in test_method:
                max_lenght_contour = max(contours, key=len)
                max_area = cv2.contourArea(max_lenght_contour)
            else:
                max_area = max(contours, key=cv2.contourArea)
        for contour in contours:
            area_contour = cv2.contourArea(contour)
            if test_method == 'max' and area_contour >= area_coeff * max_area:
                tested_contours.append(contour)
        return tested_contours


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


    #   #   #   #   #   #   #   #   #   #   #   #
    # MENU
    def keyAction(self, key):
        """
        :param key:
            e: toggles enclosing rectangles
            y: toggles canny's edges
            h: toggles high filter for edges detection
            p: increases morphological operations
            o: decreases morphological operations
        TODO: there is to be, for sure, a more compact way to do so
        """
        if key == ord('h'):
            cv2.destroyAllWindows()
            self.toggle_canny = False
            self.toggle_hipass = True
            self.toggle_inner=False
            self.resetStatistics()
        elif key == ord('y'):
            cv2.destroyAllWindows()
            self.toggle_canny = True
            self.toggle_hipass = False
            self.toggle_inner=False
            self.resetStatistics()
        elif key == ord('i'):
            cv2.destroyAllWindows()
            self.toggle_canny = False
            self.toggle_hipass = False
            self.toggle_inner=True
            self.resetStatistics()
        elif key == ord('p'):
            self.toggle_prerefinement=not self.toggle_prerefinement
        elif key == ord('e'):
            self.toggle_enclosing=not self.toggle_enclosing


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

    preprocessor = ObjectDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        exit(0)
    rospy.loginfo("exiting...")

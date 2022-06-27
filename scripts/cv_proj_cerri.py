#!/usr/bin/env python3
"""
AUTHOR:     FRANCESCO CERRI
            francesco.cerri2@studio.unibo.it

Condensed version, ROS-free version of
https://github.com/alma-x/cv_preproc

please, refer to that repository for the complete analysis over the word done, comments & improving possibilitis,
 complete code and everything else mentioned on the paper, included:
(background remotion) otsu, histogram balancing, distance & adaptive filters, timing statistics, visualizatin tools,
                        depth-based sky filtering,...
(object detector) depth based analysis, blob refinements tests, contours area thresolding, watershed separation alg.

It also inlcudes an outline of the competition, simulation setups and what ROS is

---
WHAT THIS CODE DOES

given an input from a camera mounted on a robot, helps identifying objects on the ground that can pose a risk to
robot's navigation, or are interesting in some way due to diversity from raw terrain

0) load input (video from a simulatied environment)
1) addition of noise
2) frame subsampling and noise reduction
3) (back)ground remotion based on its particular color and refinement
4) identification, refinement and highlighting of foreground objects
"""
import cv2
import numpy as np
from lib_filters import bilateralFilter, denoisingFilter,highPassFilter, medianFilter
from lib_histograms import computeHistogram
from glob import glob
import time


class ImageNoiseSimulator:
    """
    addition of noise to simulated camera view
    """
    def __init__(self,image):

        self.noise_type='ug'
        self.NOISE="gaussian, uniform"
        self.noise_intensity=4

        self.sim_img = np.ndarray
        self.in_dtype = None
        self.camera_resolution = tuple
        self.camera_channels=int
        self.noisy_img = np.ndarray

        self.initNoise(image)

    def initNoise(self,image):
        self.in_dtype = image.dtype
        self.camera_resolution = image.shape[:2]
        self.camera_channels=image.shape[2]
        print("-- NOISE SIMULATOR --\ninput resolution: {}\nsimulated noise: {}\n\n"
              .format(self.camera_resolution,self.NOISE))

    #   #   #   #   #   #   #   #
    # MAIN LOOP
    def cameraLoop(self,image):
        try:
            # SIMULATED IMAGE AQUISITION
            self.sim_img = image
            # NOISE ADDITION
            self.noiseSimualtion(
                noise_type=self.noise_type, intensity_coeff=self.noise_intensity)
            self.noisy_img = cv2.cvtColor(self.noisy_img, cv2.COLOR_RGB2BGR)
        except :pass

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    # NOISE
    def noiseSimualtion(self, noise_type=None,intensity_coeff=1):
        """
        simulate and adds to the input stream noise
        :param noise_type: 'uniform','u'; 'gaussian','g'; impulsive 'snp','i'
            NOTE: permitted multiple at once: e.g. noise_type='ug' adds both
        :param intensity_coeff: multiplicative coeff for noise intensity
        :return:
        """
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
        self.noisy_img=image

#####################################################################################
#####################################################################################

class ImagePreprocessor:
    """
    image subsampling and denoising
    for navigation camera
    """
    def __init__(self,image,downsample_coeff):

        self.noisy_img = np.ndarray
        self.in_dtype = None
        self.camera_resolution = tuple
        self.camera_channels=int
        self.current_resolution= tuple
        self.preproc_img = np.ndarray

        self.FILTERS="denoising kernel convolution -> bilateral filter"
        self.DOWNSAMPLE_COEFF=downsample_coeff
        self.initPreprocessor(image)

    def initPreprocessor(self,image):
        self.in_dtype = image.dtype
        self.camera_resolution = image.shape[:2]
        self.camera_channels=image.shape[2]
        self.noisy_img=image
        self.preproc_img=image
        self.current_resolution=self.camera_resolution[0]//self.DOWNSAMPLE_COEFF,\
                                self.camera_resolution[1]//self.DOWNSAMPLE_COEFF
        print("-- IMAGE PREPROCESSOR --\neffective resolution {}\nused filters: {}\n\n"
              .format(self.current_resolution,self.FILTERS))

    #   #   #   #   #   #   #   #
    # MAIN LOOP
    def preprocessLoop(self,image):
        try:
            # NOISY IMAGE AQUISITION
            self.noisy_img = image
            # DOWNSAMPLING: RESIZING AND BLURRING
            self.downsampleCamera(do_blur=True)
            # NOISE REMOTION
            self.noiseRemotion(do_denoise=True)
            # OUTPUT
            self.preproc_img = cv2.cvtColor(self.preproc_img, cv2.COLOR_RGB2BGR)
        except: pass

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    # IMAGE PREPROCESSING
    def downsampleCamera(self,do_blur=True):
        """
        resizes (reduces) input camera image stream self.sim_img by 1/downsize_coeff
        by default, it also blurs
        :param do_blur: if True, in addition it blurs the image accordingly,
            creating an higher level in the image pyramid

        :internal param downsize_coeff>1
        self.camera_resolution: original, self.current_resolution: resized
        """
        downsample_coeff=self.DOWNSAMPLE_COEFF
        new_size=self.camera_resolution[1]//downsample_coeff,self.camera_resolution[0]//downsample_coeff
        if do_blur and not downsample_coeff==1:
            image=self.noisy_img.copy()
            self.noisy_img = cv2.pyrDown(image, dstsize=new_size)
        else:
            image=self.noisy_img.copy()
            self.noisy_img=cv2.resize(image,new_size,interpolation=cv2.INTER_AREA)


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

        SELECTED: 1; good mix between speed, thin H hist, edges' sharpness
        """
        if do_denoise:
            image = self.noisy_img.copy()
            denoise_image = bilateralFilter(denoisingFilter(image), k_size=5, sigma_color=45, sigma_space=45)
            self.preproc_img = denoise_image.astype(self.in_dtype)
        else:
            self.preproc_img = self.noisy_img.copy().astype(self.in_dtype)

#####################################################################################
#####################################################################################

class GroundFilter:
    """
    ground remotion from image
    """
    def __init__(self,image,effective_resolution):

        self.preproc_img = np.ndarray
        self.in_dtype = None
        self.camera_channels=int
        self.current_resolution= tuple
        self.filtered_img=np.ndarray
        self.sky_mask=np.ndarray
        self.MENU_IMAGE=np.ndarray
        self.res_sample= np.ndarray
        self.res_range= np.ndarray
        self.sample_mask= np.ndarray
        # self.sample_mask=None
        self.range_mask= np.ndarray
        self.samples=[]

        self.toggle_sample=True
        self.toggle_range=False
        self.select_new_sample=False
        self.sample_source='../media/video/samples/'
        self.addedAllSample=False
        self.FILTER=""
        self.SHOW_RESULT=False
        self.selected_premask='c'
        # available premasks:   c   color
        #                       d   depth
        self.selected_filter='s'
        # available filters:    s   sample
        #                       r   range
        self.MORPH_OPS='1'#=='c'
        self.initFilter(image,effective_resolution)

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    def initFilter(self,image,effective_resolution):
        self.in_dtype = image.dtype
        self.camera_channels=image.shape[2]
        self.current_resolution = effective_resolution
        self.sky_mask=np.zeros(effective_resolution)
        self.addedAllSample=True
        self.loadAllSamples(self.sample_source)
        print("-- GROUND FILTER --\ndefault settings:\tsample filter\n\t\t\tmorph ops level 1/6\n\t\t\t"
              "only color based sky filter available\nSAMPLE FOLDER: {}".format(self.sample_source))
        print("\n_CONTROLS:\tEsc: exit\n\t\ts: displays sample filter\n\t\tr: displays range filter\n\t\t"
              "z: enables all samples for sample filter\n\t\tx: disables all samples\n\t\tl: add new sample(s) to "
              "filter from ROI(s) selection;\n\t\t\tmultiple allowed, Esc to stop selection\n\t\tk: same as 'l' & new "
              "samples are saved\n\t\tm: increase (+1/6) morphological operations amount\n\t\tn: decrease (-1/6) "
              "morphological operations amount\n\n")

#    ##################################################################################
    # MAIN LOOP
    def filterLoop(self,image):
        self.preproc_img=image

        # BACKGROUND FILTERING -------------------------------------------------------------------------
        self.SHOW_RESULT=True
        self.preMask(sky_mask_method=self.selected_premask)
        if self.toggle_sample:
            self.FILTER="GROUND SAMPLE FILTER\n                   NUMBER OF SAMPLES: {}".format(len(self.samples))
            self.sampleFilter(show_result=self.SHOW_RESULT, morph_ops=self.MORPH_OPS)
            self.filtered_img=self.res_sample.copy()
        elif self.toggle_range:
            self.FILTER="COLOR InRANGE FILTER"
            self.rangeFilter(show_result=self.SHOW_RESULT, morph_ops=self.MORPH_OPS)
            self.filtered_img=self.res_range.copy()

        # OUTPUT ------------------------------------------------------------------------------------
        self.filtered_img=cv2.cvtColor(self.filtered_img,cv2.COLOR_RGB2BGR)

        ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
        k = cv2.pollKey() & 0xff
        # k = cv2.waitKey(1) & 0xff
        if k!=255:#ANY (OTHER) KEY PRESSED
            self.keyAction(k)

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

                    -- NOT AVAILABLE IN THIS SETUP: --
                    -"depth","d": uses depth channel
                        PRO&CON:    viceversa as above
                        +CON:       also for objects w/ distance <MIN_SENSOR_RANGE
        """
        if 'c'in sky_mask_method or sky_mask_method=='color':
            image = self.preproc_img
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            #sky filtering
            blue_min=(105, 0, 0)
            blue_max=(135, 255, 255)
            self.sky_mask=cv2.inRange(hsv_image,blue_min,blue_max)


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
        """
        image = self.preproc_img
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if do_shift:
            red_max = (30, 255, 255)
            red_min = (0, 35, 50)
            red_mask = cv2.inRange(hsv_image, red_min, red_max)
            winname_prefix=winname_prefix+' shifted'
        else:
            red_max = (15, 255, 255)
            red_zeromax = (0, 30, 30)
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
            cv2.imshow('filtered image',cv2.cvtColor(self.res_range,cv2.COLOR_RGB2BGR))


    def sampleFilter(self,show_result=False, winname_prefix='', morph_ops=' '):
        """
        filter camera stream by histogram backprojection of selected samples
         builds a probability map of a certain pixel of the target img to be part of
         the sample; a treshold defines the binary mask
CV_WINDOW_AUTOSIZE
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

        NOTE: not using histogram comparison to reject samples "too dissimilar to ground"
        """
        image = self.preproc_img
        if self.select_new_sample:
            self.select_new_sample = False
            self.addNewSample()
        if self.samples:#==if not self.samples==[]
            hsv_image=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2HSV)
            _,_,image_hist=computeHistogram(hsv_image,'hs',range_normalize=True)
            sample_mask=np.zeros(np.asarray(self.current_resolution),dtype=np.uint8)
            for sample_hist in self.samples:
                add_mask=cv2.calcBackProject([hsv_image],[0,1],sample_hist,[0,180,0,256],1)
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cv2.filter2D(add_mask.copy(), -1, kernel, add_mask)
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
            cv2.imshow('filtered image',cv2.cvtColor(self.res_sample, cv2.COLOR_RGB2BGR))

    #       #       #       #       #       #       #       #
    # FILTER REFINEMENT     #       #       #
    @staticmethod
    def maskRefinement(mask,morph_ops=' ',k_size=5,k_shape=cv2.MORPH_ELLIPSE):
        """
        refine the mask w/ morphological operations; has hardcoded levels that deal greatly with the setup

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
        else:
            self.addedAllSample=False
            print('No salmple found in selected folder')

    def addNewSample(self,save_sample=False,compute_avg=False):
        """
        manual sample selection from image from a static window

        NOTE: no test over histogram correlation; ALSO samples are "instantaneous", not averaged, hence prone to noise
        YET: denoised input

        :param save_sample: toggles sample save to a file in sample directory
        :param compute_avg: computes the average color instead of adding the sample
        """
        winname="MOUSE+ENTER for selection,ESC to exit"
        img=self.preproc_img
        if not compute_avg:
            rois=cv2.selectROIs(winname,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.destroyWindow(winname)
            for (roi_x0,roi_y0,roi_x,roi_y) in rois:
                try:
                    new_sampl=img[roi_y0:roi_y+roi_y0,roi_x0:roi_x+roi_x0]
                    new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_RGB2HSV)
                    _,_,sample_hist=computeHistogram(new_sample,'hs',range_normalize=True)
                    sample_hist=sample_hist.astype('float32')
                    self.samples.append(sample_hist)
                except Exception as exc:
                    print('sample aquisition aborted '+str(exc)[13])
                if save_sample:
                    filename=self.sample_source+'sample_{}.jpg'.format(str(time.time()))
                    try:
                        cv2.imwrite(filename,cv2.cvtColor(new_sampl,cv2.COLOR_RGB2BGR))
                        print('saved: {}'.format(filename))
                    except Exception as exc:
                        print('sample saving unsuccessfull '+str(exc)[13])
        else:
            roi=cv2.selectROI(winname,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.destroyWindow(winname)
            roi_x0,roi_y0,roi_x,roi_y= roi
            new_sampl=img[roi_y0:roi_y+roi_y0,roi_x0:roi_x+roi_x0]
            new_sample=cv2.cvtColor(new_sampl,cv2.COLOR_RGB2HSV)
            arr = np.reshape(new_sample, (new_sample.shape[0] * new_sample.shape[1], 3))
            print('HSV avg: {}'+format(np.mean(arr,axis=0)))

    def removeAllSamples(self):
        """
        removes all samples in self.sample array, resetting the filter
        """
        self.samples=[]

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
            z: enables all samples in  sample folder for respective filter
            x: disables all samples
            l: add new sample(s) to filter from ROI(s) selection;
                multiple allowed, esc to exit
            i: same as l, computing average     -- UNAVAILABLE
            k: same as 'k' & new samples are saved in ./media/samples folder
            m: increase morphological operations amount
            n: decrease morphological operations amount
            d: select depth based (sky) prefilter       -- UNAVAILABLE
            c: select color based (sky) prefilter       -- ONLY ONE AVAILABLE
        """
        if key ==ord('r'):
            self.toggle_sample = False
            self.toggle_range = True
        elif key == ord('s'):
            self.toggle_range = False
            self.toggle_sample = True
        elif key == ord('m'):
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
            self.loadAllSamples()
        elif key == ord('x') and len(self.samples)>0:
            self.addedAllSample=False
            self.removeAllSamples()
        elif key == ord('k'):
            self.addNewSample(save_sample=True)
        elif key == ord('l'):
            self.addNewSample()

###############################################################################
###############################################################################

class ObjectDetector:
    """
    edges, contours and objects from the filtered image
    """
    def __init__(self,image,effective_resolution):

        self.filtered_img = np.ndarray
        self.filtered_grey = np.ndarray
        self.in_dtype = None
        self.camera_channels = int
        self.current_resolution = tuple

        self.toggle_prerefinement=True
        self.toggle_enclosing = False
        self.toggle_canny = False
        self.toggle_hipass = False
        self.toggle_inner=True
        self.edges = np.ndarray
        self.contours = []
        self.cnt_img=np.ndarray

        # text for stats window
        self.DETECTOR=""
        self.RECTANGLES=""
        self.PRE=""

        self.initDetector(image,effective_resolution)

    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    def initDetector(self,image,effective_resolution):
        self.filtered_img = image
        self.in_dtype = image.dtype
        self.camera_channels = image.shape[2]
        self.current_resolution = effective_resolution
        print("--OBJECT DETECTOR--\ndefault settings:\tinner contours from morph ops\n\t\t\topening->closing blobs "
              "prefiltering\n\n_CONTROLS:\te: toggles enclosing rectangles (GREEN rectangular contours)\n\t\ty:"
              " toggles canny's edges\n\t\th: toggles high pass filter for edges detection\n\t\ti: toggles inner edges "
              "from dilated-original blobs\n\t\tp: toggles blobs prerefinement for canny (opening->closing")

    ######    ##################################################################################
    # MAIN LOOP
    def detectorLoop(self, image,original_image):
        if self.toggle_enclosing:self.RECTANGLES="ENABLED"
        else:                   self.RECTANGLES="DISABLED"

        # PREPROCESSING --------------------------------------------------------------
        self.filtered_img=image
        grey_image = self.filtered_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
                self.PRE="ENABLED"
                # cv2.imshow('blobs', blob_img)
            else: self.PRE="DISABLED"

            self.DETECTOR="INNER, MORPH BASED"

        # EDGES -------------------------------------------------------------------------
            STRUCTURING_ELEMENT=cv2.getStructuringElement(cv2.MORPH_RECT,ksize=(3,3))
            self.edges = (cv2.dilate(blob_img.copy(),STRUCTURING_ELEMENT))-blob_img

        elif self.toggle_canny:
            window_name = "canny's"
            if self.toggle_prerefinement:
                grey_image=medianFilter(grey_image.copy(),3)
                # almost no effect compared to median filter
                # blob_img=self.objectRefinement(blob_img,'o',k_size=3)
                # blob_img=self.objectRefinement(blob_img,'c',k_size=5)
                self.PRE="ENABLED"
            else: self.PRE="DISABLED"

            # no perceived difference with changing params
            self.DETECTOR="BLOB BASED CANNY'S"
            canny_thr_high = 100
            canny_thr_low = 20
            k_size_sobel = 5
            use_L2_gradient = True
            self.edges = cv2.Canny(grey_image, canny_thr_high, canny_thr_low,
                                       apertureSize=k_size_sobel, L2gradient=use_L2_gradient)

        elif self.toggle_hipass:
            self.DETECTOR = "GREY IMG. HIGHPASS FILTER"
            # reliabily spots the inner square of the marker
            # BUT lots of internal edges at low intensity
            hipass_edges = highPassFilter(grey_image)
            hipass_edges[hipass_edges > 0] = 255
            self.edges = hipass_edges

        # CLEANING -------------------------------------------------------------------------
        # best if done before edges
        # self.edges=medianFilter(self.edges,5)

        # CLOSED CONTOURS -------------------------------------------------------------------------
        self.contours = cv2.findContours(self.edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]


        self.contours = sorted(self.contours, key=lambda ctr: cv2.contourArea(ctr), reverse=True)
        big_contours = self.contours[:20]
        rectangular_contours = self.testShape(self.contours, 4)
        rectangular_contours = sorted(rectangular_contours, key=lambda ctr: cv2.contourArea(ctr), reverse=True)
        big_rectangular_contours = rectangular_contours[:10]

        # OBJECTS -------------------------------------------------------------------------
        RGB_PURPLE = (190, 20, 100)
        RGB_CYAN = (0, 255, 255)
        RGB_YELLOW = (255, 255, 0)
        RGB_GREEN = (0, 255, 0)
        cnt_img = original_image
        cv2.drawContours(cnt_img, self.contours, -1, RGB_PURPLE, 2)
        cv2.drawContours(cnt_img, rectangular_contours, -1, RGB_YELLOW, 2)
        if self.toggle_enclosing:
            for contour in big_rectangular_contours:
                x, yy, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cnt_img, (x, yy), (x + w, yy + h), RGB_GREEN, 3)
            for contour in big_contours:
                x, yy, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cnt_img, (x, yy), (x + w, yy + h), RGB_CYAN, 2)
        cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_RGB2BGR)
        self.cnt_img=cnt_img
        cv2.imshow('contours', cnt_img)

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
            if (test_method == 'equals' and len(approx_contour) == n_edges) \
                    or (test_method == 'atleast' and len(approx_contour) >= n_edges):
                tested_contours.append(contour)
        return tested_contours

    #       #       #       #       #       #       #       #
    # REFINEMENT     #       #       #
    @staticmethod
    def objectRefinement(mask, morph_ops=' ', k_size=5, k_shape=cv2.MORPH_ELLIPSE):
        """
        refine the mask w/ morphological operations

        :param mask: input mask to be refined
        :param morph_ops: ORDERED string of required morph. operations

        :param k_size: kernel characteristic size
        :param k_shape: kernel shape:cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE}
        """
        kernel = cv2.getStructuringElement(k_shape, (k_size, k_size))
        for mop in morph_ops:
            if mop == 'd' or mop == 'dilate':
                mask = cv2.dilate(mask.copy(), kernel, iterations=1)
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
            i: inner edges, morph based over dilated-original image
            p: increases morphological operations
            o: decreases morphological operations
        """
        if key == ord('h'):
            self.toggle_canny = False
            self.toggle_hipass = True
            self.toggle_inner=False
        elif key == ord('y'):
            self.toggle_canny = True
            self.toggle_hipass = False
            self.toggle_inner=False
        elif key == ord('i'):
            self.toggle_canny = False
            self.toggle_hipass = False
            self.toggle_inner=True
        elif key == ord('p'):
            self.toggle_prerefinement=not self.toggle_prerefinement
        elif key == ord('e'):
            self.toggle_enclosing = not self.toggle_enclosing

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
if __name__ == '__main__':
    video_source="../media/video/tour.mp4"
    print("\nAUTHOR: FRANCESCO CERRI matr. 951972\n\nbackground remotion and foreground highlighting\n"
          "on a simulation of MarsYard terrain for the European Rover Challenge\n\n")

    cap=cv2.VideoCapture(video_source)
    if cap.isOpened():
        video_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame=cap.read()
        frame_resolution=frame.shape[:2]
        downsample_ratio=2
        RESIZE_COEFF=1.8
        downsampled_resolution=frame_resolution[0]//downsample_ratio,frame_resolution[1]//downsample_ratio
        cv2.namedWindow('filtered image',cv2.WINDOW_NORMAL)
        cv2.namedWindow('contours',cv2.WINDOW_NORMAL)
        cv2.namedWindow('CURRENT STATs',cv2.WINDOW_NORMAL)
        cv2.moveWindow('contours',600,0)
        cv2.moveWindow('filtered image',0,0)
        cv2.moveWindow('CURRENT STATs',0,400)
        #cv2.moveWindow('contours',int(downsampled_resolution[1]//RESIZE_COEFF),200)
        #cv2.moveWindow('CURRENT STATs',int(downsampled_resolution[1]//RESIZE_COEFF*2),int(downsampled_resolution[0]//RESIZE_COEFF)+350)
        simulator = ImageNoiseSimulator(frame)
        preprocessor=ImagePreprocessor(frame,downsample_ratio)
        ground_filter=GroundFilter(frame,downsampled_resolution)
        detector=ObjectDetector(frame,downsampled_resolution)
        while not (ret and frame is None):
            # simulated camera stream
            ret, frame = cap.read()
            # noise addition
            simulator.cameraLoop(frame)
            noisy=cv2.cvtColor(simulator.noisy_img,cv2.COLOR_RGB2BGR)
            # subsampling and noise remotion
            preprocessor.preprocessLoop(noisy)
            # (back)ground remotion
            ground_filter.filterLoop(preprocessor.preproc_img)
            new_disp_size=ground_filter.filtered_img.shape[0]//RESIZE_COEFF,ground_filter.filtered_img.shape[1]//RESIZE_COEFF
            new_disp_size=int(new_disp_size[0]),int(new_disp_size[1])
            cv2.resizeWindow('filtered image',new_disp_size[1],new_disp_size[0])
            # objects identification
            detector.detectorLoop(ground_filter.filtered_img,preprocessor.preproc_img)
            new_disp_size=detector.cnt_img.shape[0]//RESIZE_COEFF,detector.cnt_img.shape[1]//RESIZE_COEFF
            new_disp_size=int(new_disp_size[0]),int(new_disp_size[1])
            cv2.resizeWindow('contours',new_disp_size[1],new_disp_size[0])
            # stats window
            stats_image=np.zeros_like(preprocessor.preproc_img)
            STATS_TEXT ="SELECTED FILTER(s/r): {}\n\nBACKGROUND REMOTION MORPH. OPs LEVEL(n/m): {}/6\n\n" \
                        "OBJ.DETECTION METHOD(i/y/h): {}\n\nENCLOSING RECTANGLES(e): {}\n\nOBJECT PREFILTERING(p): {}"\
                        "\n\nEsc to exit must be shortly hold, aswell as e and i/h/y"\
                .format(ground_filter.FILTER,ground_filter.MORPH_OPS,detector.DETECTOR,detector.RECTANGLES,detector.PRE)
            y0, dy = 30, cv2.getTextSize(STATS_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 1, 4)[0][1]
            # text splitting and multi line printing
            for ii, line in enumerate(STATS_TEXT.split('\n')):
                y = y0 + ii * int(2 * dy)
                cv2.putText(stats_image, line, (20, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            cv2.imshow('CURRENT STATs',stats_image)
            new_disp_size=stats_image.shape[0]//RESIZE_COEFF,stats_image.shape[1]//RESIZE_COEFF
            new_disp_size=int(new_disp_size[0]*1.2),int(new_disp_size[1]*1.2)
            cv2.resizeWindow('CURRENT STATs',new_disp_size[1],new_disp_size[0])
            
            k = cv2.pollKey() & 0xff
            # k = cv2.waitKey(1) & 0xff
            if k == 27: print('exiting');cv2.destroyAllWindows();break
            elif k==45:RESIZE_COEFF+=.2
            elif k==43 and RESIZE_COEFF>1: RESIZE_COEFF-=.2
            elif k != 255:
                ground_filter.keyAction(k)
                detector.keyAction(k)
            # restar video when it terminates
            if cap.get(cv2.CAP_PROP_POS_FRAMES)==video_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,1)

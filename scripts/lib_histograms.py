#!/usr/bin/env python3

import cv2
import numpy as np

def computeHistogram(image, color_chan, flat=False, mask=None, accumulate=False, range_normalize=False):
    # TODO: automatically convert colorspace wrt color_chan; NEEDS specific input img
    # TODO img flatten+ hist faster? np.histogram faster?
    #  hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    channel_dict = {'r': {'size': [256], 'range': [0, 256], 'channel': [0]},
                    'g': {'size': [256], 'range': [0, 256], 'channel': [1]},
                    'b': {'size': [256], 'range': [0, 256], 'channel': [2]},
                    'rgb': {'size': [256], 'range': [0, 256], 'channel': [0, 1, 2]},
                    'h': {'size': [180], 'range': [0, 180], 'channel': [0]},
                    's': {'size': [256], 'range': [0, 256], 'channel': [1]},
                    'v': {'size': [256], 'range': [0, 256], 'channel': [2]},
                    'hs': {'size': [180, 256], 'range': [0,180,0, 256], 'channel': [0, 1]},
                    'gl': {'size': [256], 'range': [0, 256], 'channel': [0]}
                    }
    chan_data = channel_dict[color_chan]
    if flat:
        image = image.flatten()
    # if 'h'in color_chan or 's'in color_chan or 'v'in color_chan:
    #     image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([image], chan_data['channel'], mask, chan_data['size'], chan_data['range'],
                        accumulate=accumulate)
    nhist = np.ndarray.copy(hist) / hist.sum()
    if range_normalize:
        rhist = np.ndarray.copy(hist)
        cv2.normalize(rhist, rhist, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
        rhist = np.int32(np.around(rhist))
        return hist, nhist, rhist
    return hist, nhist

def drawHistogram(histogram, img_zoom=1, shift_percent=0, man_scale=1.0, bar_style=False, base_img=None):
    # TODO: add support of base img rather than blank one
    #  needing obviously to assert correctness of dimensions
    """
    draw an histogram passed as input, letting modify img resolution,
    toward-right shifting, scaling of the plot

    default img size: 256*256
    default style: polyline
    :param histogram: must be an histogram (eg output of cv2.calcHist
    :param img_zoom: final img size *=img_zoom
    :param shift_percent: % shift toward right
    :param man_scale: vertical plot scaling
    :param bar_style: toggles bar plot instead of polyline
    :param base_img: if not None, uses a non black image as base
    :return: image
    """
    hist_size = histogram.shape[0]
    bins_array = np.arange(hist_size, dtype=np.int32).reshape(hist_size, 1)
    if img_zoom != 1:
        bins_array *= img_zoom
    hist_height = 256
    hist_img = np.zeros((img_zoom * hist_height, img_zoom * hist_size))
    # SHIFT TO RIGHT; ENHANCES VISIBILITY OF RED PEAK
    shift_amount = int(hist_size * shift_percent / 100)
    histogram = np.roll(histogram, shift_amount)
    # MANUAL SCALING; CONSERVES PROPORTIONS IN MULTI PLOT IMAGES
    if man_scale != 1:
        histogram = np.int32(np.around(man_scale * histogram))
    # TODO: PLOT STYLE
    # if bar_style:
    #     bin_w = hist_size*img_zoom
    #     for x, y in enumerate(histogram):
    #         cv2.rectangle(hist_height,(x*bin_w,y[0]),((x+1)*bin_w-1,hist_height),255,-1)
    #     return np.flipud(hist_img)
    # polyline style
    points = np.column_stack((bins_array, histogram))
    cv2.polylines(hist_img, [points], False, [255],thickness=2)
    return np.flipud(hist_img)


# OLD showHistogram from image_filter.ImagePreprocNode
# serving as backup
# def showHistograms(self,channels,noisy='n',winname_prefix=''):
# # def showHistograms(self, noisy='n', winname_prefix=''):
#     # TODO: decide qhich channel to compute and show based on detection of letter in kw
#     # TODO: change noise param to toggle superimposing of noisy histogram
#     #  to the sim one
#     """
#     computes and shows H,S and RGB histograms
#     :return: none
#     """
#     rgb_scaling = 0.01
#     hsv_shift = 15
#     rgb_shift=92
#
#     if noisy == 'n' or noisy == 'noisy':
#         winname_prefix = winname_prefix + ' NOISY '
#         rgb_in_image = self.noisy_img
#     elif noisy == 'd' or noisy == 'denoised':
#         rgb_in_image = self.dnoise_img
#         winname_prefix = winname_prefix + ' DENOISED '
#     else:
#         rgb_in_image = self.sim_img
    #
    # hsv_in_image = cv2.cvtColor(rgb_in_image.copy(), cv2.COLOR_RGB2HSV)
    #
    # _, _, h_hist = computeHistogram(hsv_in_image, 'h', range_normalize=True)
    # # _, _, s_hist = computeHistogram(hsv_in_image, 's', range_normalize=True)
    # # _, _, v_hist = computeHistogram(hsv_in_image, 'v', range_normalize=True)
    # h_hist_img = drawHistogram(h_hist, img_zoom=3, shift_percent=hsv_shift)
    # # s_hist_img= drawHistogram(s_hist,img_zoom=3,shift_percent=hsv_shift)
    # # ACTUALLY IT IS BGR
    # # rgb_hist_img = cv2.merge([b_hist_img, g_hist_img, r_hist_img])
    # shift_text = ", SHIFT AMOUNT= " + str(int(hs_shift * 1.8)) + " /180"
    # cv2.imshow(winname_prefix + "H HIST" + shift_text, h_hist_img)
    # # cv2.imshow(winname_prefix+"S HIST"+shift_text, s_hist_img)
    # # shift_text=", SHIFT AMOUNT= " + str(int(rgb_shift*2.56)) + " /256"
    # # cv2.imshow(winname_prefix+"RGB HIST"+shift_text, rgb_hist_img)
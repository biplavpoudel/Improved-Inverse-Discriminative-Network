# Implementation of Otsu's Algorithm for Binarization of bimodal, grayscale Signature Image
# histogram is used to detect optimal (global) threshold
# that separates two regions: background and foreground, with maximum inter-class variance

import numpy as np
import cv2 as cv


def Otsu(image, is_normalized ):
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)

    revised_image = image - threshold
    cv.imshow('revised image', revised_image)
    cv.waitKey(0)


if __name__ == '__main__':
    image = cv.imread(
        r'D:\MLProjects\Inverse-Discriminative-Network\dataset_process\CEDAR\signatures\full_forg\forgeries_1_1.png')
    Otsu(image, is_normalized=True)

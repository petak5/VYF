import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
import skimage
from PIL import Image
import numpy as np
from skimage.transform import pyramid_gaussian, pyramid_laplacian
import argparse
import cv2 as cv

def main():
    args = getArgs()

    print("Loading images...")

    image_t = skimage.io.imread(args.target)
    image_s = skimage.io.imread(args.source)
    print(image_t.shape)
    img_t = rgb2gray(image_t)
    img_s = rgb2gray(image_s)

    print("Done")
    print("Detecting descriptors of target image...")

    keypoints_t, descriptors_t = extract_features(img_t)

    print("Done")
    print("Detecting descriptors of source image...")

    keypoints_s, descriptors_s = extract_features(img_s)

    print("Done")
    print("Matching descriptors...")

    matches = match_descriptors(descriptors_s, descriptors_t, max_ratio=0.6, cross_check=True)

    print("Done")

    show_matches(img_s, img_t, keypoints_s, keypoints_t, matches)

    print("Descriptors 1: ", descriptors_s.size)
    print("Descriptors 2: ", descriptors_t.size)
    print("Keypoints 1: ", keypoints_s.size)
    print("Keypoints 2: ", keypoints_t.size)
    print("Matches: ", matches.size)

    # TODO: Improve this
    keypoints_s_filtered = []
    keypoints_t_filtered = []
    deltas = []
    for match in matches:
            k_s = keypoints_s[match[0]]
            k_t = keypoints_t[match[1]]

            keypoints_s_filtered.append([k_s[1], k_s[0]])
            keypoints_t_filtered.append([k_t[1], k_t[0]])

            dx = k_s[0] - k_t[0]
            dy = k_s[1] - k_t[1]

            deltas.append([dx, dy])

    keypoints_s_filtered = np.array(keypoints_s_filtered)
    keypoints_t_filtered = np.array(keypoints_t_filtered)
    deltas = np.array(deltas)
    median = np.median(deltas, axis=0)

    ####################
    # Primitive method #
    ####################
    dx = median[1]
    dy = median[0]
    vector = (dx, dy)
    print("Translation vector: ", vector)
    tform_align = transform.AffineTransform(translation=vector)
    img_s_aligned = transform.warp(image_s, tform_align, mode="edge", preserve_range=True).astype(image_s.dtype)

    img_diff_orig = skimage.util.compare_images(image_t, image_s)
    img_diff_aligned = skimage.util.compare_images(image_t, img_s_aligned)
    plt.imshow(img_diff_orig)
    plt.title("No Alignment")
    plt.show()
    plt.imshow(img_diff_aligned)
    plt.title("Primitive Alignment")
    plt.show()

    skimage.io.imsave("./test_orig.png", skimage.util.img_as_ubyte(img_diff_orig))
    skimage.io.imsave("./test_aligned.png", skimage.util.img_as_ubyte(img_diff_aligned))

    ##################
    # Proper methods #
    ##################
    homography = compute_homography(keypoints_s_filtered, keypoints_t_filtered)
    print("Homography:\n", homography)
    img_s_aligned_opencv = align_opencv(image_s, homography)
    img_s_aligned_skimage = align_skimage(image_s, homography)
    skimage.io.imsave("./test_aligned_opencv.png", skimage.util.img_as_ubyte(img_s_aligned_opencv))
    skimage.io.imsave("./test_aligned_skimage.png", skimage.util.img_as_ubyte(img_s_aligned_skimage))

    img_diff_aligned_opencv = skimage.util.compare_images(image_t, img_s_aligned_opencv)
    img_diff_aligned_skimage = skimage.util.compare_images(image_t, img_s_aligned_skimage)
    skimage.io.imsave("./test_diff_aligned_opencv.png", skimage.util.img_as_ubyte(img_diff_aligned_opencv))
    skimage.io.imsave("./test_diff_aligned_skimage.png", skimage.util.img_as_ubyte(img_diff_aligned_skimage))

    plt.imshow(img_diff_aligned_opencv)
    plt.title("Warped with calculated homography\nusing matched features from SIFT (OpenCV)")
    plt.show()
    plt.imshow(img_diff_aligned_skimage)
    plt.title("Warped with calculated homography\nusing matched features from SIFT (skimage)")
    plt.show()

    # orb_detector = cv.ORB_create()
    # orb_detector

def getArgs():
    parser = argparse.ArgumentParser(prog='ImageRegistration',
                            description='Image registration for HDR image creation.',
                            epilog='Text at the bottom of help')
    parser.add_argument("target")
    parser.add_argument("source")

    return parser.parse_args()

def extract_features(image):
    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(image)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    return keypoints, descriptors

def show_matches(image_source, image_target, keypoints_source, keypoints_target, matches):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(13, 10))

    plt.gray()

    plot_matches(ax[0], image_source, image_target, keypoints_source, keypoints_target, matches)
    ax[0].axis('off')
    ax[0].set_title("Source Image and Target Image\n"
                    "(all keypoints and matches)")

    plot_matches(ax[1], image_source, image_target, keypoints_source, keypoints_target, matches[::15],
            only_matches=True)
    ax[1].axis('off')
    ax[1].set_title("Source Image and Target Image\n(subset of matches for visibility)")

    plt.tight_layout()
    plt.show()

def compute_homography(keypoints_source, keypoints_target):
    assert(keypoints_source.shape[0] >= 4)
    assert(keypoints_target.shape[0] >= 4)
    h = cv.findHomography(srcPoints=keypoints_source, dstPoints=keypoints_target)#, method=cv.RANSAC, ransacReprojThreshold=5.0)
    return h[0]

def align_opencv(image_source, homography):
    return cv.warpPerspective(image_source, homography, [image_source.shape[1], image_source.shape[0]])

def align_skimage(image_source, homography):
    return cv.warpPerspective(image_source, homography, [image_source.shape[1], image_source.shape[0]])

main()

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
from datetime import datetime
from math import floor, ceil

RESIZE_IMAGE = True
RESIZE_RATIO = 4

def main():
    args = getArgs()

    print("Loading images...")

    # image_t = skimage.io.imread(args.target)
    # image_s = skimage.io.imread(args.source)
    image_t = cv.imread(args.target)
    image_s = cv.imread(args.source)

    print("Done")
    print("Original size: ", image_t.shape)
    print("Adjusting images for SIFT")

    img_t = rgb2gray(image_t)
    img_s = rgb2gray(image_s)

    img_t = skimage.exposure.equalize_hist(img_t)
    img_t = skimage.util.img_as_ubyte(img_t)

    img_s = skimage.exposure.equalize_hist(img_s)
    img_s = skimage.util.img_as_ubyte(img_s)

    # Resize image used for SIFT
    if RESIZE_IMAGE:
        img_t = skimage.transform.resize(img_t, (img_t.shape[0] // RESIZE_RATIO, img_t.shape[1] // RESIZE_RATIO), anti_aliasing=True)
        img_t = skimage.util.img_as_ubyte(img_t)
        img_s = skimage.transform.resize(img_s, (img_s.shape[0] // RESIZE_RATIO, img_s.shape[1] // RESIZE_RATIO), anti_aliasing=True)
        img_s = skimage.util.img_as_ubyte(img_s)

        print("Resized size: ", img_t.shape)

    print("Done")

    #################
    # Proper method #
    #################

    print("Initializing SIFT...")
    sift = cv.SIFT_create()
    print("Done")
    print("Detecting features in source image...")
    kp_s, des_s = sift.detectAndCompute(img_s, None)
    print("Done")
    print("Detecting features in target image...")
    kp_t, des_t = sift.detectAndCompute(img_t, None)
    print("Done")
    print("Source features: ", len(kp_s))
    print("Target features:", len(kp_t))

    print("Matching features...")
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_s, des_t, k=2)

    print("Done")
    print("Matches: ", len(matches))

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ tuple(x * RESIZE_RATIO for x in kp_s[m.queryIdx].pt) for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ tuple(x * RESIZE_RATIO for x in kp_t[m.trainIdx].pt) for m in good ]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # print(kp_s[0].pt * RESIZE_RATIO)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    print("Drawing matches...")

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img_matches = cv.drawMatches(img_s,kp_s,img_t,kp_t,good,None,**draw_params)
    # plt.imshow(img_matches, 'gray'),plt.show()

    print("Done")

    aligned_s = align_opencv(image_s, M)

    cropped_s = crop_image(aligned_s, M)
    cropped_t = crop_image(image_t, M)

    # img_diff = skimage.util.compare_images(cropped_s, image_s)
    img_diff_1 = cv.absdiff(cropped_s, cropped_t)
    img_diff_2 = 255 - img_diff_1

    cv.imwrite("./cropped_s.png", cropped_s)
    cv.imwrite("./cropped_t.png", cropped_t)

    cv.imwrite("./diff_1.png", img_diff_1)
    cv.imwrite("./diff_2.png", img_diff_2)

    print("Done")

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

def align_opencv(image, homography):
    return cv.warpPerspective(image, homography, [image.shape[1], image.shape[0]])

def align_skimage(image, homography):
    return cv.warpPerspective(image, homography, [image.shape[1], image.shape[0]])

# Crop image according to the homography transform
def crop_image(image, homography):
    h,w = image.shape[0:2]

    points = np.array([[0,0], [0, h-1], [w-1, 0], [w-1, h-1]], dtype=float).reshape(-1, 1, 2)
    points_transformed = cv.perspectiveTransform(points, homography)

    # For some reason every point is embedded in another list, that's why there is [0] in the middle of every indexing
    left_x = ceil(max(max(0, points_transformed[0][0][0]), points_transformed[1][0][0]))
    right_x = floor(min(min(w, points_transformed[2][0][0]), points_transformed[3][0][0]))
    top_y = ceil(max(max(0, points_transformed[0][0][1]), points_transformed[2][0][1]))
    bottom_y = floor(min(min(h, points_transformed[1][0][1]), points_transformed[3][0][1]))

    from_right = w - right_x
    from_bottom = h - bottom_y

    return skimage.util.crop(image, ((top_y, from_bottom), (left_x, from_right), (0, 0)), copy=False)

def old_skimage_method():
    # print("Detecting descriptors of target image...")

    # start=datetime.now()

    # keypoints_t, descriptors_t = extract_features(img_t)

    # print("Done")
    # print("Detecting descriptors of source image...")

    # keypoints_s, descriptors_s = extract_features(img_s)

    # print("Done")
    # print("Matching descriptors...")

    # matches = match_descriptors(descriptors_s, descriptors_t, max_ratio=0.6, cross_check=True)

    # print("Done")
    # print(datetime.now()-start)

    # show_matches(img_s, img_t, keypoints_s, keypoints_t, matches)

    # print("Source descriptors: ", descriptors_s.size)
    # print("Target descriptors: ", descriptors_t.size)
    # print("Source keypoints: ", keypoints_s.size)
    # print("Target keypoints: ", keypoints_t.size)
    # print("Matches: ", matches.size)

    # # TODO: Improve this
    # keypoints_s_filtered = []
    # keypoints_t_filtered = []
    # deltas = []
    # for match in matches:
    #         k_s = keypoints_s[match[0]]
    #         k_t = keypoints_t[match[1]]

    #         keypoints_s_filtered.append([k_s[1], k_s[0]])
    #         keypoints_t_filtered.append([k_t[1], k_t[0]])

    #         dx = k_s[0] - k_t[0]
    #         dy = k_s[1] - k_t[1]

    #         deltas.append([dx, dy])

    # keypoints_s_filtered = np.array(keypoints_s_filtered)
    # keypoints_t_filtered = np.array(keypoints_t_filtered)
    # deltas = np.array(deltas)
    # median = np.median(deltas, axis=0)

    # ####################
    # # Primitive method #
    # ####################
    # dx = median[1]
    # dy = median[0]
    # vector = (dx, dy)
    # print("Translation vector: ", vector)
    # tform_align = transform.AffineTransform(translation=vector)
    # img_s_aligned = transform.warp(image_s, tform_align, mode="edge", preserve_range=True).astype(image_s.dtype)

    # img_diff_orig = skimage.util.compare_images(image_t, image_s)
    # img_diff_aligned = skimage.util.compare_images(image_t, img_s_aligned)
    # plt.imshow(img_diff_orig)
    # plt.title("No Alignment")
    # plt.show()
    # plt.imshow(img_diff_aligned)
    # plt.title("Primitive Alignment")
    # plt.show()

    # skimage.io.imsave("./test_orig.png", skimage.util.img_as_ubyte(img_diff_orig))
    # skimage.io.imsave("./test_aligned.png", skimage.util.img_as_ubyte(img_diff_aligned))

    #################
    # Proper method #
    #################

    # homography = compute_homography(keypoints_s_filtered, keypoints_t_filtered)
    # print("Homography:\n", homography)
    # img_s_aligned_opencv = align_opencv(image_s, homography)
    # img_s_aligned_skimage = align_skimage(image_s, homography)
    # skimage.io.imsave("./test_aligned_opencv.png", skimage.util.img_as_ubyte(img_s_aligned_opencv))
    # skimage.io.imsave("./test_aligned_skimage.png", skimage.util.img_as_ubyte(img_s_aligned_skimage))

    # img_diff_aligned_opencv = skimage.util.compare_images(image_t, img_s_aligned_opencv)
    # img_diff_aligned_skimage = skimage.util.compare_images(image_t, img_s_aligned_skimage)
    # skimage.io.imsave("./test_diff_aligned_opencv.png", skimage.util.img_as_ubyte(img_diff_aligned_opencv))
    # skimage.io.imsave("./test_diff_aligned_skimage.png", skimage.util.img_as_ubyte(img_diff_aligned_skimage))

    # plt.imshow(img_diff_aligned_opencv)
    # plt.title("Warped with calculated homography\nusing matched features from SIFT (OpenCV)")
    # plt.show()
    # plt.imshow(img_diff_aligned_skimage)
    # plt.title("Warped with calculated homography\nusing matched features from SIFT (skimage)")
    # plt.show()

    raise NotImplementedError

main()

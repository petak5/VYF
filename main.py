import matplotlib.pyplot as plt
from pathlib import Path
from skimage.color import rgb2gray
import skimage
import numpy as np
import argparse
import cv2 as cv
from math import floor, ceil

RESIZE_IMAGE = True
RESIZE_RATIO = 4

def main():
    args = getArgs()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.output + "/diff/").mkdir(parents=True, exist_ok=True)
    Path(args.output + "/SIFT/").mkdir(parents=True, exist_ok=True)
    Path(args.output + "/result/").mkdir(parents=True, exist_ok=True)

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
        return 1

    print("Drawing matches...")

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img_keypoints_s = cv.drawKeypoints(img_s, kp_s, 0)
    img_keypoints_t = cv.drawKeypoints(img_t, kp_t, 0)
    # _ = plt.figure("Source keypoints")
    # plt.imshow(img_keypoints_s, 'gray')
    # _ = plt.figure("Target keypoints")
    # plt.imshow(img_keypoints_t, 'gray')
    # plt.show()

    img_matches = cv.drawMatches(img_s,kp_s,img_t,kp_t,good,None,**draw_params)
    # plt.imshow(img_matches, 'gray')
    # plt.show()

    cv.imwrite(f"{args.output}/SIFT/keypoints_s.png", img_keypoints_s)
    cv.imwrite(f"{args.output}/SIFT/keypoints_t.png", img_keypoints_t)
    cv.imwrite(f"{args.output}/SIFT/matches.png", img_matches)

    print("Done")

    # Cropped and aligned
    uncropped_aligned_s = align_image(image_s, M)
    cropped_aligned_s = crop_image(uncropped_aligned_s, M)
    cropped_t = crop_image(image_t, M)

    cv.imwrite(f"{args.output}/result/uncropped_target.png", image_t)
    cv.imwrite(f"{args.output}/result/uncropped_aligned_s.png", uncropped_aligned_s)

    cv.imwrite(f"{args.output}/result/cropped_s.png", cropped_aligned_s)
    cv.imwrite(f"{args.output}/result/cropped_t.png", cropped_t)

    # Extended and aligned images
    extended_aligned_s, extended_t = align_and_resize_images(image_s, image_t, M)

    cv.imwrite(f"{args.output}/result/extended_aligned_s.png", extended_aligned_s)
    cv.imwrite(f"{args.output}/result/extended_t.png", extended_t)

    # Diff images
    orig_diff = cv.absdiff(image_s, image_t)
    cropped_diff_1 = cv.absdiff(cropped_aligned_s, cropped_t)
    cropped_diff_2 = 255 - cropped_diff_1
    extended_diff = cv.absdiff(extended_aligned_s, extended_t)

    cv.imwrite(f"{args.output}/diff/diff_orig.png", orig_diff)
    cv.imwrite(f"{args.output}/diff/cropped_diff_1.png", cropped_diff_1)
    cv.imwrite(f"{args.output}/diff/cropped_diff_2.png", cropped_diff_2)
    cv.imwrite(f"{args.output}/diff/extended_diff.png", extended_diff)

    print("Done")

    return 0

def getArgs():
    parser = argparse.ArgumentParser(prog='ImageRegistration',
                            description='Image registration for HDR image creation.',
                            epilog='Text at the bottom of help.')
    parser.add_argument("-t", "--target", required=True, help="Target image.")
    parser.add_argument("-s", "--source", required=True, help="Source image.")
    parser.add_argument("-o", "--output", default="./out/", help="Output directory.")

    return parser.parse_args()

def align_image(image, homography):
    return cv.warpPerspective(image, homography, [image.shape[1], image.shape[0]])

def align_and_resize_images(image_s, image_t, homography):
    """ Extend and align images

    Source image is resized to fit it's alignment warping, target image is also resized to the same dimensions.
    """
    h,w = image_s.shape[0:2]
    points = np.array([[0,0], [0, h-1], [w-1, 0], [w-1, h-1]], dtype=float).reshape(-1, 1, 2)
    points_transformed = cv.perspectiveTransform(points, homography)

    top = ceil(abs(0 + min(0, points_transformed[0][0][1], points_transformed[1][0][1], points_transformed[2][0][1], points_transformed[3][0][1])))
    bottom = ceil(abs(h - max(h, points_transformed[0][0][1], points_transformed[1][0][1], points_transformed[2][0][1], points_transformed[3][0][1])))
    left = ceil(abs(0 - min(0, points_transformed[0][0][0], points_transformed[1][0][0], points_transformed[2][0][0], points_transformed[3][0][0])))
    right = ceil(abs(w - max(w, points_transformed[0][0][0], points_transformed[1][0][0], points_transformed[2][0][0], points_transformed[3][0][0])))

    extended_image = cv.copyMakeBorder(image_s, top, bottom, left, right, cv.BORDER_CONSTANT, None, value=0)

    points += np.array([left, top])
    points_transformed += np.array([left, top])

    new_homography = cv.findHomography(points, points_transformed, cv.RANSAC, 5.0)[0]

    aligned_image = cv.warpPerspective(extended_image, new_homography, [w + left + right, h + top + bottom])
    extended_t = cv.copyMakeBorder(image_t, top, bottom, left, right, cv.BORDER_CONSTANT, None, value=0)
    return aligned_image, extended_t

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

if __name__ == "__main__":
    exit(main())

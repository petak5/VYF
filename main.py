import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
import skimage
from PIL import Image
import numpy as np

# im = Image.open("./DSC_0148.jpeg")
# im.show()
# pix = np.array(im, dtype=np.uint8)
# img = rgb2gray(pix)
# plt.imshow(pix)
# plt.show()

print("Loading images...")

image1 = skimage.io.imread("./Images/DSC_0148.jpg")
image2 = skimage.io.imread("./Images/DSC_0149.jpg")
# image1 = skimage.io.imread("./Images/DSC_0067.jpg")
# image2 = skimage.io.imread("./Images/DSC_0068.jpg")
print(image1.shape)
img1 = rgb2gray(image1)
img2 = rgb2gray(image2)

print("Done")
print("Detecting descriptors of image 1...")

descriptor_extractor = SIFT()

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

print("Done")
print("Detecting descriptors of image 2...")

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

print("Done")
print("Matching descriptors...")

matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6, cross_check=True)

print("Done")

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(13, 10))

plt.gray()

plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Flipped Image\n"
                   "(all keypoints and matches)")

plot_matches(ax[1], img1, img2, keypoints1, keypoints2, matches12[::15],
             only_matches=True)
ax[1].axis('off')
ax[1].set_title("Original Image vs. Flipped Image\n(subset of matches for visibility)")

plt.tight_layout()
plt.show()

print("Descriptors 1: ", descriptors1.size)
print("Descriptors 2: ", descriptors2.size)
print("Keypoints 1: ", keypoints1.size)
print("Keypoints 2: ", keypoints2.size)
print("Matches: ", matches12.size)

deltas = []
for match in matches12:
       k1 = keypoints1[match[0]]
       k2 = keypoints2[match[1]]

       dx = k1[0] - k2[0]
       dy = k1[1] - k2[1]

       deltas.append([dx, dy])

temp2 = np.array(deltas)
median = np.median(temp2, axis=0)
# print(median)

dx = -median[1]
dy = -median[0]
vector = (dx, dy)
tform_align = transform.AffineTransform(translation=vector)
img2_aligned = transform.warp(image2, tform_align, mode="edge", preserve_range=True).astype(image2.dtype)

img_diff_orig = skimage.util.compare_images(image1, image2)
img_diff_aligned = skimage.util.compare_images(image1, img2_aligned)
plt.imshow(img_diff_orig)
plt.show()
plt.imshow(img_diff_aligned)
plt.show()

skimage.io.imsave("./test_orig.png", skimage.util.img_as_ubyte(img_diff_orig))
skimage.io.imsave("./test_aligned.png", skimage.util.img_as_ubyte(img_diff_aligned))

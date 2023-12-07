import numpy as np
import cv2
from feature_extraction import mean_luminance, extract_glcm_features, lbp
from sklearn import svm
from skimage.segmentation import slic, mark_boundaries


'''
Based on "Image Colorization Method Using Texture
Descriptors and ISLIC Segmentation" by Cao Liqin,
Lei Jiao, and Zhijiang Li (March 2017)
https://www.researchgate.net/publication/315468487_Image_Colorization_Method_Using_Texture_Descriptors_and_ISLIC_Segmentation

Colorization portion based on 
Welsh, T., Ashikhmin, M., Mueller, K. (2002): Transferring color to greyscale images. 
ACMTransactions on Graphics (TOG), 21(3), 277-280
https://www.researchgate.net/publication/220183710_Transferring_Color_to_Greyscale_Images 

Image datasets from: https://github.com/ByUnal/Example-based-Image-Colorization-w-KNN
'''

#Step 0 Load in Reference and Target Image as grayscale
ref = cv2.imread('../data/p014_a_source.png', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('../data/p014_b_target.png', cv2.IMREAD_GRAYSCALE)
print(ref.shape)
print(target.shape)
#show images
cv2.imshow('Reference', ref)
cv2.imshow('Target', target)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Step 1 Feature Extraction (Luminance, Entropy, Homogeneity, Correlation, LBP, and SVM)
target_luminance = mean_luminance(target)
target_entropy, target_homogeneity, target_correlation = extract_glcm_features(target)
target_lbp = lbp(target)

ref_luminance = mean_luminance(ref)
ref_entropy, ref_homogeneity, ref_correlation = extract_glcm_features(ref)
ref_lbp = lbp(ref)

print(target_luminance.shape)
print(target_entropy.shape)
print(target_homogeneity.shape)
print(target_correlation.shape)
print(target_lbp.shape)

# Step 2 Superpixel Extraction Using ISLIC for grayscale
num_segments = 800
alpha = 0.8
beta = 0.8
segments_target = slic(target, n_segments=num_segments, compactness=alpha, sigma=beta, channel_axis=None)
segments_ref = slic(ref, n_segments=num_segments, compactness=alpha, sigma=beta, channel_axis=None)
target_superpixels = mark_boundaries(target, segments_target)
ref_superpixels = mark_boundaries(ref, segments_ref)

cv2.imshow('Target Superpixels', target_superpixels)
cv2.imshow('Reference Superpixels', ref_superpixels)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3 Feature Mapping and Colorization


# Welsh et al steps for colorization:
# 
# 1. We loop through each superpixel in the target image
#    For each superpixel in target,
#       a. find the best matching superpixel in the reference image
#          using difference of luminance, entropy, homogeneity, correlation,
#          and LBP between target and reference superpixels
#       b. Once the best matching superpixel is found, convert both
#         superpixels to LAB color space
#      c. Iterate through each pixel of the target superpixel and
#         calculate neighborhood statistics to find best matching pixel
#         within the reference superpixel
#           - sample 50 random pixels from the target superpixel
#           - calculate 5x5 neighborhood average luminance and standard deviation
#             of the luminance values
#           - Decide on best match for the target pixel by average of
#             luminance and standard deviation
#      d. Assign the a,b channel of the best matching pixel in the reference to 
#         the target pixel

def superpixel_features(image, segments):
    for i in np.max(segments) + 1: 
        for row in image.shape[0]:
            for col in image.shape[1]:
                if segments[row, col] == i:
                    pass

def find_best_match(target, ref):
    pass


num_superpixels_target = np.max(segments_target) + 1

for i in num_superpixels_target: # for each super pixel
    for row in target.shape[0]:
        for col in target.shape[1]:
            if segments_target[row, col] == i: 

                # find the best matching superpixel in the reference image for the target
                best_match = find_best_match(target, ref)
                
                # convert both su

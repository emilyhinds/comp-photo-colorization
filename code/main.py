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

Image datasets from: https://github.com/ByUnal/Example-based-Image-Colorization-w-KNN
'''

#Step 0 Load in Reference and Target Image as grayscale
ref = cv2.imread('../castle.jpeg', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('../castle.jpeg', cv2.IMREAD_GRAYSCALE)
print(ref.shape)
print(target.shape)
#show images
cv2.imshow('Reference', ref)
cv2.waitKey(0)
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
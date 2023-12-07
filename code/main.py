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


# # Step 1 Feature Extraction (Luminance, Entropy, Homogeneity, Correlation, LBP, and SVM)
# target_luminance = mean_luminance(target)
# target_entropy, target_homogeneity, target_correlation = extract_glcm_features(target)
# target_lbp = lbp(target)

# ref_luminance = mean_luminance(ref)
# ref_entropy, ref_homogeneity, ref_correlation = extract_glcm_features(ref)
# ref_lbp = lbp(ref)

# print(target_luminance.shape)
# print(target_entropy.shape)
# print(target_homogeneity.shape)
# print(target_correlation.shape)
# print(target_lbp.shape)

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
    '''
    Calculates entropy and mean luminence of each superpixel
    and puts them into dictionaries where their superpixel number
    functions as the key
    '''
    superpixel_features = {}
    for i in range(np.max(segments) + 1): 
        superpixel = []
        min_row = np.inf
        max_row = -np.inf
        min_col = np.inf
        max_col = -np.inf
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if segments[row, col] == i:
                    if image[row, col] == 0:
                        superpixel.append(0.000000001)
                    else:
                        superpixel.append(image[row, col])
                    if row < min_row:
                        min_row = row
                    if row > max_row:
                        max_row = row
                    if col < min_col:
                        min_col = col
                    if col > max_col:
                        max_col = col

                    #ensure max and min row and max and min col are two away from edge
                    if min_row < 2:
                        min_row = 2
                    if min_col < 2:
                        min_col = 2
                    if max_row > image.shape[0] - 2:
                        max_row = image.shape[0] - 2
                    if max_col > image.shape[1] - 2:
                        max_col = image.shape[1] - 2

                    
        superpixel = np.array(superpixel)
        

        mean_luminance = np.mean(superpixel)
        entropy = -np.sum(superpixel * np.log(superpixel))
        superpixel_features[i] = (mean_luminance, entropy, min_row, max_row, min_col, max_col)

    return superpixel_features


reference_features  = superpixel_features(ref, segments_ref)
target_features = superpixel_features(target, segments_target)


def find_best_match(target_features, ref_superpixels):
    best_diff = np.inf
    best_match = None
    print(len(ref_superpixels))
    for i in range(len(ref_superpixels)):
        difference  = np.abs(target_features[0] - ref_superpixels[i][0]) + np.abs(target_features[1] - ref_superpixels[i][1])
        if difference < best_diff:
            best_diff = difference
            best_match = i
    
    print(best_match)
    return best_match

    



num_superpixels_target = np.max(segments_target) + 1

def colorize(target, ref, segments_target, segments_ref, reference_features, target_features):
    '''
    Colorizes the target image using the reference image
    '''
    for row in range (2, target.shape[0]-2):
        for col in range(2, target.shape[1]-2):

            #get best superpixel match
            target_superpixel = segments_target[row, col]
            target_superpixel_features = target_features[target_superpixel]
            ref_superpixel = find_best_match(target_superpixel_features, reference_features)
            
            #get best pixel match in superpixel
            target_pixel = target[row, col]
            neighbors = target[row-2:row+2, col-2:col+2][0]
            target_measure = np.abs(((0.5 * np.mean(neighbors))  + (0.5 * np.std(neighbors))) / 2)

            best_match = None
            best_diff = np.inf

            #find best match in reference superpixel
            random_pixel = 0
            while (random_pixel < 50):
                rand_row = np.random.randint(reference_features[ref_superpixel][2], reference_features[ref_superpixel][3])
                rand_col = np.random.randint(reference_features[ref_superpixel][4], reference_features[ref_superpixel][5]) 
                if segments_ref[rand_row, rand_col] == ref_superpixel:
                    ref_pixel = ref[rand_row, rand_col]
                    ref_neighbors = ref[rand_row-2:rand_row+2, rand_col-2:rand_col+2][0]
                    ref_measure = np.abs(((0.5 * np.mean(ref_neighbors))  + (0.5 * np.std(ref_neighbors))) / 2)
                    difference = np.abs(target_measure - ref_measure)
                    if difference < best_diff:
                        best_diff = difference
                        best_match = ref_pixel
                    random_pixel += 1

            #assign best match to target pixel
            target[row, col] = best_match
    
    return target


colorized = colorize(target, ref, segments_target, segments_ref, reference_features, target_features)

#Convert back form LAB to RGB
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
cv2.imshow('Colorized', colorized)


            
                                                         


                        
            


# for i in num_superpixels_target: # for each super pixel
#         for row in target.shape[0]:
#             for col in target.shape[1]:
#                 if segments_target[row, col] == i: 
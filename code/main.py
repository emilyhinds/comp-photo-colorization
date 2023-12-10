import numpy as np
import cv2
from feature_extraction import mean_luminance, extract_glcm_features, lbp
from sklearn import svm
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm
import matplotlib.pyplot as plt


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
curr_img = 7

ref = cv2.imread('../data/p00' + str(curr_img) + '_a_source.png', cv2.IMREAD_GRAYSCALE)
ref_color = cv2.imread('../data/p00' + str(curr_img) + '_a_source.png', cv2.IMREAD_COLOR)
target = cv2.imread('../data/p00' + str(curr_img) + '_b_target.png', cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread('../data/p00' + str(curr_img) + '_c_groundtruth.png', cv2.IMREAD_COLOR)


print(ref.shape)
print(target.shape)
#show images
cv2.imshow('Target', target)
cv2.imshow('Reference', ref)
cv2.imshow('Reference Color', ref_color)

cv2.waitKey()
# cv2.destroyAllWindows()


# Step 2 Superpixel Extraction Using ISLIC for grayscale


# num_segments = 800
num_segments = 200
alpha = 0.8
beta = 0.8
segments_target = slic(target, n_segments=num_segments, compactness=alpha, sigma=beta, channel_axis=None)
segments_ref = slic(ref, n_segments=num_segments, compactness=alpha, sigma=beta, channel_axis=None)
target_superpixels = mark_boundaries(target, segments_target)
ref_superpixels = mark_boundaries(ref, segments_ref)

cv2.imshow('Target Superpixels', target_superpixels)
cv2.imshow('Reference Superpixels', ref_superpixels)
cv2.waitKey(0)
# cv2.destroyAllWindows()

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
                    superpixel.append(image[row, col] + 0.0000000000000001)
                    if row < min_row:
                        min_row = row
                    if row > max_row:
                        max_row = row
                    if col < min_col:
                        min_col = col
                    if col > max_col:
                        max_col = col

                    # ensure max and min row and max and min col are two away from edge
                    if min_row < 2:
                        min_row = 2
                    if min_col < 2:
                        min_col = 2
                    if max_row > image.shape[0] - 2:
                        max_row = image.shape[0] - 2
                    if max_col > image.shape[1] - 2:
                        max_col = image.shape[1] - 2
                    
        superpixel = np.array(superpixel).flatten()
        mean_luminance = np.mean(superpixel)
        entropy = -np.sum(superpixel * np.log(superpixel))
        superpixel_features[i] = (mean_luminance, entropy, min_row, max_row, min_col, max_col)

    return superpixel_features

def show_superpixel(superpixel_id, segments, image):
    '''
    Shows the superpixel with the given superpixel number
    '''
    print("show superpixel")
    print(image.dtype)
    print(image)
    superpixel_image = np.zeros((segments.shape[0], segments.shape[1]))
    for row in range(segments.shape[0]):
        for col in range(segments.shape[1]):
            if segments[row, col] == superpixel_id:
                superpixel_image[row, col] = image[row, col]
    cv2.imshow('Superpixel', superpixel_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find_best_match(target_features, ref_superpixels):
    best_diff = np.inf
    best_match = None
    for i in range(len(ref_superpixels)):
        difference  = np.abs(target_features[0] - ref_superpixels[i][0]) + np.abs(target_features[1] - ref_superpixels[i][1])
        if difference < best_diff:
            best_diff = difference
            best_match = i
 
    return best_match


def colorize(target, ref_color, segments_target, segments_ref, reference_features, target_features):
    '''
    Colorizes the target image using the reference image
    '''
    print(target.dtype)
    print(ref_color.dtype)
    target = (target/256).astype(np.float32) # Convert target to float32
    ref_color = cv2.cvtColor((ref_color/256).astype(np.float32), cv2.COLOR_BGR2Lab) # Convert color reference to LAB space
    # print(target) # goes 0 -> 1
    # print(ref_color) # goes -128 -> 127
    print("color ref type:", ref_color.dtype)
    print("target type:", target.dtype)
    # ref_color = ref_color.astype(np.float32)

    colorized = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.float32)
    
    # loop through target image except along edges
    for row in tqdm(range(2, target.shape[0]-2)):
        for col in range(2, target.shape[1]-2):
            #print(ref_color[row][col])

            #get best superpixel match
            target_superpixel = segments_target[row, col]
            target_superpixel_features = target_features[target_superpixel]
            ref_superpixel = find_best_match(target_superpixel_features, reference_features)
            
            # show_superpixel(target_superpixel, segments_target, target)
            # print(ref_color.shape)
            # show_superpixel(ref_superpixel, segments_ref, ref_color[:,:,0])


            #get best pixel match in superpixel
            target_pixel = target[row, col]
            neighbors = target[row - 2 : row + 2, col - 2 : col + 2]
            target_measure = np.abs(((0.5 * np.mean(neighbors))  + (0.5 * np.std(neighbors))) / 2)

            best_match = None
            best_diff = np.inf

            #find best match in reference superpixel
            random_pixel = 0
            while (random_pixel < 50):
                rand_row = np.random.randint(reference_features[ref_superpixel][2], reference_features[ref_superpixel][3])
                rand_col = np.random.randint(reference_features[ref_superpixel][4], reference_features[ref_superpixel][5]) 
                if segments_ref[rand_row, rand_col] == ref_superpixel:
                    ref_pixel = ref_color[rand_row, rand_col]
                    ref_neighbors = ref_color[rand_row - 2 : rand_row + 2, rand_col - 2 : rand_col + 2][0]
                    ref_measure = np.abs(((0.5 * np.mean(ref_neighbors))  + (0.5 * np.std(ref_neighbors))) / 2)
                    difference = np.abs(target_measure - ref_measure)
                    if difference < best_diff:
                        best_diff = difference
                        best_match = ref_pixel
                    random_pixel += 1

            # print(best_match)
            #assign best match to target pixel
            #print(best_match)
            colorized[row, col] = (target_pixel, best_match[1], best_match[2])
            # print(colorized[row, col])
    
    return colorized

def colorize2(target, ref):
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
    colorized = np.zeros((target.shape[0], target.shape[1], 3))
    for row in tqdm(range(2, target.shape[0]-2)):
        for col in range(2, target.shape[1]-2):
            
            
            # print(row/target.shape[0])
            
            target_pixel = target[row, col]
            
            neighbors = target[row-2:row+2, col-2:col+2][0]
            target_measure = np.abs(((0.5 * np.mean(neighbors))  + (0.5 * np.std(neighbors))) / 2)

            best_match = None
            best_diff = np.inf

            for _ in range(200):
                rand_row = np.random.randint(2, ref.shape[0]-2)
                rand_col = np.random.randint(2, ref.shape[1]-2)
                ref_pixel = ref[rand_row, rand_col]
                ref_neighbors = ref[rand_row-2:rand_row+2, rand_col-2:rand_col+2][0]
                ref_measure = np.abs(((0.5 * np.mean(ref_neighbors))  + (0.5 * np.std(ref_neighbors))) / 2)
                difference = np.abs(target_measure - ref_measure)
                if difference < best_diff:
                    best_diff = difference
                    best_match = ref_pixel

            colorized[row, col] = (target_pixel, best_match[1], best_match[2])

    return colorized


reference_features  = superpixel_features(ref, segments_ref)
target_features = superpixel_features(target, segments_target)

colorized = colorize(target, ref_color, segments_target, segments_ref, reference_features, target_features)
# Split the LAB image into its channels
l_channel, a_channel, b_channel = cv2.split(colorized)

# Display each channel separately using cv2.imshow
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(l_channel, cmap='gray')
plt.title('L Channel')

plt.subplot(1, 3, 2)
plt.imshow(a_channel, cmap='gray')
plt.title('A Channel')

plt.subplot(1, 3, 3)
plt.imshow(b_channel, cmap='gray')
plt.title('B Channel')

plt.show()
# colorized_without_border = colorized[2:colorized.shape[0]-2, 2:colorized.shape[1]-2, :]
print(colorized.dtype)
# print(colorized[2:colorized.shape[0]-2, 2:colorized.shape[1]-2, :])


# colorized = colorize2(target, ref_color)
colorized[:,:,0] = colorized[:,:,0] * 256
colorized[:,:,1] = colorized[:,:,1] + 128
colorized[:,:,2] = colorized[:,:,2] + 128
print(colorized[2:colorized.shape[0]-2, 2:colorized.shape[1]-2, :])

colorized = (colorized).astype(np.uint8)

 


#Convert back form LAB to RGB
colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)

print("POST CONVERSION TO BGR")
print(colorized[2:colorized.shape[0]-2, 2:colorized.shape[1]-2, :])


colorized = (colorized).astype(np.uint8)

print("POST CONVERSION TO UINT8")
print(colorized[2:colorized.shape[0]-2, 2:colorized.shape[1]-2, :])

cv2.imshow('Ground Truth', ground_truth)
cv2.imshow('Colorized', colorized)
cv2.waitKey()
cv2.destroyAllWindows()

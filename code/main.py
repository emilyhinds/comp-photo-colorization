import numpy as np
import cv2
from sklearn import svm
from skimage.segmentation import slic, mark_boundaries
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import lab2rgb, rgb2lab
from skimage import io
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
curr_img = 9

ref = cv2.imread('../data/' + str(curr_img) + '_a_source.png', cv2.IMREAD_GRAYSCALE)
ref_color = cv2.imread('../data/' + str(curr_img) + '_a_source.png', cv2.IMREAD_COLOR)
target = cv2.imread('../data/' + str(curr_img) + '_b_target.png', cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread('../data/' + str(curr_img) + '_c_groundtruth.png', cv2.IMREAD_COLOR)


# print(ref.shape)
# print(target.shape)
# show images
cv2.imshow('Target', target)
cv2.imshow('Reference', ref)
cv2.imshow('Reference Color', ref_color)

cv2.waitKey(0)
# cv2.destroyAllWindows()


# Step 2 Superpixel Extraction Using ISLIC for grayscale

# segment the image and then run the svm on each superpixel
# for naive classification using a radial basis kernel function in the SVM


# num_segments = 800 # as suggested by paper
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



def local_features(img):
    '''
    Calculates mean luminance, entropy, homogeneity, correlation, and LBP for image based 
    on sliding 3x3 window to acheive local texture descriptors
    '''
    features = np.zeros((img.shape[0]-2, img.shape[1]-2, 5))
    for row in tqdm(range(1, img.shape[0]-1)):
        for col in range(1, img.shape[1]-1):
            window = img[row-1:row+1, col-1:col+1]
            mean_luminance = np.mean(window)
            glcm = graycomatrix(window, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=False)
            entropy = np.mean(graycoprops(glcm, 'energy'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            correlation = np.mean(graycoprops(glcm, 'correlation'))
            lbp = np.mean(local_binary_pattern(window, 8, 1, method='uniform'))
            # print("mean luminance : ", mean_luminance)
            # print("entropy : ", entropy)
            # print("homogeneity : ", homogeneity)
            # print("correlation : ", correlation)
            # print("lbp : ", lbp)

            features[row - 1, col - 1] = (mean_luminance, entropy, homogeneity, correlation, lbp)
    return features

# def max_color_channel(ref, ref_color):
#     bgr_image = np.zeros_like(ref)

#     for row in range (ref.shape[0]):
#         for col in range (ref.shape[1]):
#             max_channel  = np.argmax(ref_color[row, col])
#             bgr_image[row, col] = max_channel
#             # print(max_channel)
#     return bgr_image

def max_color_channel(ref_color, c1, c2):
    max_channels = np.zeros((ref_color.shape[0], ref_color.shape[1]))
    for row in range (ref_color.shape[0]):
        for col in range (ref_color.shape[1]):
            if ref_color[row, col, c1] > ref_color[row, col, c2]:
                max_channels[row, col] = c1
            else:
                max_channels[row, col] = c2
    return max_channels

#loop through target clsses to make combined 
def voting(bg, gr, rb):
    target_class = np.zeros((bg.shape[0], bg.shape[1]))
    for row in range (target_class.shape[0]):
        for col in range (target_class.shape[1]):
            if bg[row, col] == gr[row, col]:
                target_class[row, col] = bg[row, col]
            elif bg[row, col] == rb[row, col]:
                target_class[row, col] = bg[row, col]
            elif gr[row, col] == rb[row, col]:
                target_class[row, col] = gr[row, col]

    return target_class

def visualize_classifier(classifier, description):
    '''
    Visualizes a given classifier by displaying blue, green, or red
    in each pixel depending on its classification
    '''
    visualize = np.zeros((classifier.shape[0], classifier.shape[1], 3))
    for row in range (classifier.shape[0]):
        for col in range (classifier.shape[1]):
            if classifier[row, col] == 0:
                visualize[row, col] = [255, 0, 0]
            elif classifier[row, col] == 1:
                visualize[row, col] = [0, 255, 0]
            elif classifier[row, col] == 2:
                visualize[row, col] = [0, 0, 255]
    
    cv2.imshow(description, visualize)
    cv2.waitKey(0)

    # visualize_lab = cv2.cvtColor(visualize.copy().astype(np.uint8), cv2.COLOR_BGR2Lab)
    # cv2.imshow(description, visualize)
    # cv2.imshow(description + " in Lab", visualize_lab)
    
    # visualize_bgr_again = cv2.cvtColor(visualize_lab, cv2.COLOR_Lab2BGR)
    # cv2.imshow(description + " in BGR", visualize_bgr_again)
    # cv2.waitKey(0)


print("calculating reference local features")
ref_features = local_features(ref)
# print("reference features", ref_features)
print("calculating target local features")
target_features = local_features(target)
# print("target_features", target_features)


# # Chris one-to-one approach:

# # label_color = max_color_channel(ref, ref_color)
# bg_max = max_color_channel(ref_color, 0, 1)
# gr_max = max_color_channel(ref_color, 1, 2)
# rb_max = max_color_channel(ref_color, 0, 2)
# bg_max = bg_max[1:bg_max.shape[0]-1, 1:bg_max.shape[1]-1]
# gr_max = gr_max[1:gr_max.shape[0]-1, 1:gr_max.shape[1]-1]
# rb_max = rb_max[1:rb_max.shape[0]-1, 1:rb_max.shape[1]-1]
# # label_color = label_color[1:label_color.shape[0]-1, 1:label_color.shape[1]-1]

# # Step 2.5 SVM Classification

# # svm_model = svm.SVC(kernel='rbf')
# svm_model_bg = svm.SVC(kernel='rbf')
# svm_model_gr = svm.SVC(kernel='rbf')
# svm_model_rb = svm.SVC(kernel='rbf')
# print("fitting model")
# # DOES this need to reassign variables or does calling fit change the model itself?
# #svm_model.fit(ref_features.flatten().reshape(-1, 5), label_color.flatten())
# # svm_model.fit(ref_features.flatten().reshape(-1, 5), label_color.flatten())
# svm_model_bg.fit(ref_features.flatten().reshape(-1, 5), bg_max.flatten())
# svm_model_gr.fit(ref_features.flatten().reshape(-1, 5), gr_max.flatten())
# svm_model_rb.fit(ref_features.flatten().reshape(-1, 5), rb_max.flatten())

# print("predicting target")
# # target_class = svm_model.predict(target_features.flatten().reshape(-1, 5))
# target_class_bg = svm_model_bg.predict(target_features.flatten().reshape(-1, 5))
# target_class_gr = svm_model_gr.predict(target_features.flatten().reshape(-1, 5))
# target_class_rb = svm_model_rb.predict(target_features.flatten().reshape(-1, 5))
# # target_class = target_class.reshape((target.shape[0]-2, target.shape[1]-2))
# target_class_bg = target_class_bg.reshape((target.shape[0]-2, target.shape[1]-2))
# target_class_gr = target_class_gr.reshape((target.shape[0]-2, target.shape[1]-2))
# target_class_rb = target_class_rb.reshape((target.shape[0]-2, target.shape[1]-2))

# target_class = voting(target_class_bg, target_class_gr, target_class_rb)
# label_color  = voting(bg_max, gr_max, rb_max)

# print('target class', target_class)
# print('target class shape', target_class.shape)
# visualize_classifier(target_class, 'target image classifier')

# print("predicting ref")
# ref_class = label_color
# print('ref class', ref_class)
# print('ref class shape', ref_class.shape)
# visualize_classifier(ref_class, 'reference image classifier')
# # ref_class = svm_model.predict(ref_features.flatten().reshape(-1, 5))
# # ref_class = ref_class.reshape((ref.shape[0]-2, ref.shape[1]-2))
# print("done predicting")

# EMILY FUNCTIONS

def make_label(ref_color, color_channel):
    '''
    Makes a label image for the given an m x n x 3 color image reference and color channel.
    Params:
        ref_color: an m x n x 3 color reference image
        color_channel: 0 (blue), 1 (green), or 2 (red) for the color channel to calculate
    Returns: an m x n numpy array with 1s in pixels where the given color channel has
    the highest value, 0s everywhere else
    '''
    label = np.zeros((ref_color.shape[0], ref_color.shape[1]))
    for row in range(ref_color.shape[0]):
        for col in range(ref_color.shape[1]):
            if color_channel == np.argmax(ref_color[row][col]):
                label[row][col] = 1
    
    return label

def merge_bgr_classifiers(target_b_class, target_g_class, target_r_class):
    '''
    Creates one classifier for the target image based on the separate b, g, r ones.
    When a single pixel is predicted to have multiple best matches, one is randomly
    chosen (THIS MIGHT NEED TO CHANGE FOR THE RESULTS TO LOOK BETTER - for example
    we could default to blue or somehow find out what the majority of the reference 
    image was and set that to be the default)

    Params: 
        target_b_class: 1 where pixels are predicted to have blue as the highest color channel
        target_g_class: 1 where pixels are predicted to have green as the highest color channel
        target_r_class: 1 where pixels are predicted to have red as the highest color channel
    
    Returns:
        Numpy array the size of the inputs image with pixel values corresponding to 
        the predicted color channel with the hightest value: 0, 1, 2 (blue, green, red)
    '''
    merged_class = np.empty_like(target_b_class)
    b_counter = 0
    g_counter = 0
    r_counter = 0
    bg_counter = 0
    gr_counter = 0
    rb_counter = 0
    bgr_counter = 0
    none_counter = 0

    for row in range(target_b_class.shape[0]):
        for col in range(target_b_class.shape[1]):
            b = target_b_class[row][col]
            g = target_g_class[row][col]
            r = target_r_class[row][col]

            # blue
            if b == 1 and g == 0 and r == 0: 
                merged_class[row][col] = 0
                b_counter += 1
            # green
            elif b == 0 and g == 1 and r == 0:
                merged_class[row][col] = 1
                g_counter += 1
            # red
            elif b == 0 and g == 0 and r == 1:
                merged_class[row][col] = 2
                r_counter += 1
            # tied blue and green --> blue
            elif b == 1 and g == 1 and r == 0:
                merged_class[row][col] = 0
                bg_counter += 1
            # tied green red --> green
            elif b == 0 and g == 1 and r == 1:
                merged_class[row][col] = 1
                gr_counter += 1
            # tied red blue --> red
            elif b == 1 and g == 0 and r == 1:
                merged_class[row][col] = 2
                rb_counter += 1
            # all tied (1) --> red
            elif b == 1 and g == 1 and r == 1:
                merged_class[row][col] = 2
                bgr_counter += 1
            # all tied (0) --> blue
            elif b == 0 and g == 0 and r == 0:
                merged_class[row][col] = 0
                none_counter += 1
   

    print('Blues: ', str(b_counter), ' Greens: ', str(g_counter), ' Reds: ', str(r_counter))
    print('Blue-Green ties: ', str(bg_counter), ' Green-Red ties: ', str(gr_counter), ' Red-Blue ties: ', str(rb_counter))
    print('Alls: ', str(bgr_counter), ' Nones: ', str(none_counter))
    return merged_class


# Emily new attempt: one-to-rest approach
# aim to separate one color channel from the other two
# Have b, g, r labels

# make labels
b_label = make_label(ref_color, 0)
g_label = make_label(ref_color, 1)
r_label = make_label(ref_color, 2)
# reshape (might not need to do this)
b_label = b_label[1:b_label.shape[0]-1, 1:b_label.shape[1]-1]
g_label = g_label[1:g_label.shape[0]-1, 1:g_label.shape[1]-1]
r_label = r_label[1:r_label.shape[0]-1, 1:r_label.shape[1]-1]

# create models
b_svm_model = svm.SVC(kernel='rbf')
g_svm_model = svm.SVC(kernel='rbf')
r_svm_model = svm.SVC(kernel='rbf')

# why is this being reshaped? does this need to be reassigned variables?
print("fitting blue model")
b_svm_model.fit(ref_features.flatten().reshape(-1, 5), b_label.flatten())
print("fitting green model")
g_svm_model.fit(ref_features.flatten().reshape(-1, 5), g_label.flatten())
print("fitting red model")
r_svm_model.fit(ref_features.flatten().reshape(-1, 5), r_label.flatten())

# predict
print("predicting blue classification for target")
target_b_class = b_svm_model.predict(target_features.flatten().reshape(-1, 5))
print("predicting green classification for target")
target_g_class = g_svm_model.predict(target_features.flatten().reshape(-1, 5))
print("predicting red classification for target")
target_r_class = r_svm_model.predict(target_features.flatten().reshape(-1, 5))
# reshape
target_b_class = target_b_class.reshape((target.shape[0]-2, target.shape[1]-2))
target_g_class = target_g_class.reshape((target.shape[0]-2, target.shape[1]-2))
target_r_class = target_r_class.reshape((target.shape[0]-2, target.shape[1]-2))

# now we have 3 arrays which predict whether each pixel will be blue, green, or red
# merge to create an array which predicts the color channel with the highest value
target_class = merge_bgr_classifiers(target_b_class, target_g_class, target_r_class)

# check results
print('target class shape', target_class.shape)
visualize_classifier(target_class, 'target image classifier')

# (redundant) but visualize the reference classifier
ref_class = merge_bgr_classifiers(b_label, g_label, r_label)
visualize_classifier(ref_class, 'reference image classifier')


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

def superpixel_features(image, segments, classification):
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
        b_count = 0
        g_count = 0
        r_count = 0
        b = 0
        g = 0
        r = 0
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
                    
                    if row > 0 and col > 0 and row < image.shape[0] - 1 and col < image.shape[1] - 1:
                        if classification[row-1, col-1] == 0:
                            b_count += 1
                        elif classification[row-1, col-1] == 1:
                            g_count += 1
                        elif classification[row-1, col-1] == 2:
                            r_count += 1

        if b_count >= 50:
            b = 1
        if g_count >= 50:
            g = 1
        if r_count >= 50:
            r = 1

        superpixel = np.array(superpixel).flatten()
        mean_luminance = np.mean(superpixel)
        entropy = -np.sum(superpixel * np.log(superpixel))
        superpixel_features[i] = (mean_luminance, entropy, min_row, max_row, min_col, max_col, b, g, r)

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


def find_best_match(target_features, ref_superpixels, target_pixel_class):
    '''
    Finds the best matching superpixel in the reference image
    target_features: tuple of mean luminance and entropy of target superpixel
    ref_superpixels: dictionary of superpixel features in reference image
    target_pixel_class: class of target pixel (0, 1, or 2) for b, g, or r
    ref_class: array of superpixel classes in reference image
    ref: reference image
    
    returns: best_match, an integer that is the superpixel number of the best match
             in the reference image
    '''
    best_diff = np.inf
    best_match = None
    # print('target pixel class', target_pixel_class)
    for i in range(len(ref_superpixels)):
        difference  = np.abs(target_features[0] - ref_superpixels[i][0]) + np.abs(target_features[1] - ref_superpixels[i][1])
        # print(difference) # sometimes this is nan
        if difference < best_diff:
            # print(ref_superpixels[i][6 + target_pixel_class])
            # print(ref_superpixels[i][6 + target_pixel_class] == 1)

            if target_pixel_class == 0:
                if ref_superpixels[i][6] == 1:
                    best_diff = difference
                    best_match = i
                    # print("found best match, # " + str(i) + " for class " + str(target_pixel_class))
            
            elif target_pixel_class == 1:
                if ref_superpixels[i][7] == 1:
                    best_diff = difference
                    best_match = i
                    # print("found best match, # " + str(i) + " for class " + str(target_pixel_class))
            
            elif target_pixel_class == 2:    
                if ref_superpixels[i][8] == 1:
                    best_diff = difference
                    best_match = i
                    # print("found best match, # " + str(i) + " for class " + str(target_pixel_class))

            # # this should work but I don't think it does for some reason 
            # if ref_superpixels[i][6 + target_pixel_class] == 1:
            #     best_diff = difference
            #     best_match = i
            #     # this seems to never work for classes other than 0
            #     print("found best match, # " + str(i) + " for class " + str(target_pixel_class))
 
    return best_match


def colorize(target, ref_color, segments_target, segments_ref, reference_features, target_features, ref_class, target_class):
    '''
    Colorizes the target image using the reference image
    '''
    print(target.dtype)
    print(ref_color.dtype)
    # target = (target/256).astype(np.float32) # Convert target to float32
    # ref_color = cv2.cvtColor((ref_color/256).astype(np.float32), cv2.COLOR_BGR2Lab) # Convert color reference to LAB space
    
    ref_color = cv2.cvtColor(ref_color, cv2.COLOR_BGR2Lab) # Convert color reference to LAB space
    
    # print(target) # goes 0 -> 1
    # print(ref_color) # goes -128 -> 127
    print("color ref type:", ref_color.dtype)
    print("target type:", target.dtype)
    # ref_color = ref_color.astype(np.float32)

    colorized = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    
    # no_matches = 0
    
    print('Colorizing locally')

    # loop through target image except along edges
    for row in tqdm(range(2, target.shape[0]-2)):
        for col in range(2, target.shape[1]-2):
            #print(ref_color[row][col])

            #get best superpixel match
            target_superpixel = segments_target[row, col]
            target_superpixel_features = target_features[target_superpixel]
            target_pixel_class = target_class[row-1, col-1]
            ref_superpixel = find_best_match(target_superpixel_features, reference_features, target_pixel_class)
            
            # print("DONE FINDING BEST MATCH")
           
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
                    reference_pixel_class = ref_class[rand_row-1, rand_col-1]

                    if target_pixel_class == reference_pixel_class:
                        if (difference < best_diff):
                            # print("FOUND BEST MATCH PIXEL")
                            best_diff = difference
                            best_match = ref_pixel
                        random_pixel += 1

            colorized[row, col] = (target_pixel, best_match[1], best_match[2])
            # if best_match is None:
            #     no_matches += 1
            #     colorized[row, col] = (target_pixel, 0, 0)
            # else: 
            #     colorized[row, col] = (target_pixel, best_match[1], best_match[2])
            
    # print("no matches: ", no_matches)
    return colorized
 

def display_lab(image):
    '''
    Displays the LAB channels of an image
    '''
    l_channel, a_channel, b_channel = cv2.split(image)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(l_channel, cmap='gray')
    plt.title('L Channel: luminance')

    plt.subplot(1, 3, 2)
    plt.imshow(a_channel, cmap='gray')
    plt.title('A Channel: green-red')

    plt.subplot(1, 3, 3)
    plt.imshow(b_channel, cmap='gray')
    plt.title('B Channel: blue-yellow')

    plt.show()

def neighborhoods(image):
    '''
    Returns image of same size where pixel values are tuples, average luminance and standard deviation of luminance
    '''
    neighborhoods = np.zeros((image.shape[0], image.shape[1]), dtype=object)
    for row in range(2, image.shape[0]-2):
        for col in range(2, image.shape[1]-2):
            neighbors = image[row - 2 : row + 2, col - 2 : col + 2]
            neighborhoods[row, col] = (np.mean(neighbors), np.std(neighbors))

    return neighborhoods

def colorize2(target, ref, ref_class, target_class):
    target_feats = neighborhoods(target)
    ref_feats = neighborhoods(ref)
    print("colorize 2 start", target.dtype) # uint8
    # print(target) # 0-255
    print(target.dtype) # uint8
    print(ref.dtype) # uint8
    # print(target) # 0-255
    
    ref = cv2.cvtColor((ref), cv2.COLOR_BGR2Lab) # Convert color reference to LAB space

    print(ref.dtype) # uint8
    # print(ref) # 0-255

    print('Starting global colorizing')

    colorized = np.zeros((target.shape[0], target.shape[1], 3)).astype(np.uint8)

    for row in tqdm(range(2, target.shape[0]-2)):
    # for row in range(2, target.shape[0]-2):
        for col in range(2, target.shape[1]-2):
            
            target_pix = target[row, col]
            target_feature = target_feats[row, col]
            target_pixel_class = target_class[row-1, col-1]

            best_match = None
            best_diff = np.inf

            valid_pixels = 0
            while valid_pixels < 200:
                rand_row = np.random.randint(2, ref.shape[0]-2)
                rand_col = np.random.randint(2, ref.shape[1]-2)
                ref_pixel = ref[rand_row, rand_col]
                ref_feature = ref_feats[rand_row, rand_col]
                difference  = ((np.abs(target_feature[0] - ref_feature[0])*0.5) + (np.abs(target_feature[1] - ref_feature[1])*0.5))/2
                reference_pixel_class = ref_class[rand_row-1, rand_col-1]

                if target_pixel_class == reference_pixel_class:
                    valid_pixels += 1
                    if difference < best_diff:
                        best_diff = difference
                        best_match = ref_pixel


            colorized[row, col] = (target_pix, best_match[1], best_match[2])
            # print('best match', best_match)
            # print('combined', colorized[row, col])

    # print("end of colorize2 function")
    # print(colorized.dtype)
    # print(colorized[2:colorized.shape[0]-2, 2:colorized.shape[1]-2, :])
    return colorized


reference_features  = superpixel_features(ref, segments_ref, ref_class)
target_features = superpixel_features(target, segments_target, target_class)

colorized = colorize(target, ref_color, segments_target, segments_ref, reference_features, target_features, ref_class, target_class)
colorized2 = colorize2(target, ref_color, ref_class, target_class)

display_lab(colorized)
display_lab(colorized2)

# print("post colorized")
# print(colorized.dtype)
# print(colorized[2:colorized.shape[0]-2, 2:colorized.shape[1]-2, :])
# cv2.imshow('Colorized still in Lab', colorized)
# cv2.imshow('Colorized2 still in Lab', colorized2)

#Convert back form LAB to BGR
colorized_cv2 = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized2_cv2 = cv2.cvtColor(colorized2, cv2.COLOR_Lab2BGR)
# print("POST CONVERSION TO BGR cv2")
# print(colorized_cv2.dtype)
# print(colorized_cv2[2:colorized_cv2.shape[0]-2, 2:colorized_cv2.shape[1]-2, :])


cv2.imshow('Ground Truth', ground_truth)
cv2.imshow('Local Colorized converted to bgr using cv2', colorized_cv2)
cv2.imshow('Global Colorized2 converted to bgr using cv2', colorized2_cv2)
cv2.waitKey(0)

# display the colorized results with the highest value color channel
local_b_label = make_label(colorized_cv2, 0)
local_g_label = make_label(colorized_cv2, 1)
local_r_label = make_label(colorized_cv2, 2)

local_merged = merge_bgr_classifiers(local_b_label, local_g_label, local_r_label)
visualize_classifier(local_merged, "Locally Colorized")

global_b_label = make_label(colorized_cv2, 0)
global_g_label = make_label(colorized_cv2, 1)
global_r_label = make_label(colorized_cv2, 2)

global_merged = merge_bgr_classifiers(global_b_label, global_g_label, global_r_label)
visualize_classifier(global_merged, "Globally Colorized")

cv2.waitKey(0)



# colorized_skimage = lab2rgb(colorized).astype(np.float32)
# print("POST CONVERSION TO RGB skimage as type uint8")
# print(colorized_skimage.dtype)
# print(colorized_skimage[2:colorized_skimage.shape[0]-2, 2:colorized_skimage.shape[1]-2, :])
# # show result
# io.imshow(colorized_skimage)
# io.show()

# cv2.imshow('Colorized converted to rgb using skimage', colorized_skimage)
# bgr_colorized_skimage = colorized_skimage[:, :, ::-1].copy()
# cv2.imshow('Colorized converted to bgr using skimage', bgr_colorized_skimage)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

import numpy as np
import cv2
from skimage.feature import graycomatrix, local_binary_pattern


def mean_luminance(img):
    '''
    Calculates the mean luminance of an image.
    '''
    return np.mean(img)


# def compute_glcm(image, distances, angles):
#     glcm = np.zeros((256, 256, len(distances), len(angles)), dtype=np.uint32)

#     for d, distance in enumerate(distances):
#         for a, angle in enumerate(angles):
#             dx = int(distance * np.cos(angle))
#             dy = int(distance * np.sin(angle))

#             for i in range(image.shape[0] - abs(dx)):
#                 for j in range(image.shape[1] - abs(dy)):
#                     glcm[image[i, j], image[i + dx, j + dy], d, a] += 1

#     return glcm

def glcm(img):
    '''
    Calculates the Gray Level Co-occurrence Matrix (GLCM)
    of an image. This is used to extract texture features
    from the image that can be used to characterize local
    structure. The features extracted from the GLCM are the
    entropy, homogeneity, and correlation.
    '''
    glcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True)

    return glcm


def extract_entropy(img, glcm):
    '''
    Calculates the entropy of the image using the GLCM.
    Entropy represents how homogeneous the scene is, or how
    similar the pixels are to each other in the entire image.
    '''
    normalized_glcm = glcm / np.sum(glcm)
    # replace 0s with small value to avoid log(0)
    normalized_glcm[normalized_glcm == 0] = 1e-10
    entropy = -np.sum(normalized_glcm * np.log2(normalized_glcm))

    return entropy
    

def extract_homogeneity(img, glcm):
    '''
    Calculates the homogeneity of the image using the GLCM.
    Homogeneity (aka Angular Second Moment) represents how
    homogeneous the image is.
    '''
    homogeneity = 0
    for i in range(256):
        for j in range(256):
            homogeneity += glcm[i][j] / (1 + np.abs(i - j))
    return homogeneity

def extract_correlation(img, glcm):
    '''
    Calculates the correlation of the image using the GLCM.
    Correlation represents how correlated the pixels are to
    each other in the entire image.
    '''
    correlation = 0
    for i in range(256):
        for j in range(256):
            correlation += glcm[i][j] * (i - mean_luminance(img)) * (j - mean_luminance(img)) / (np.std(img) ** 2)
    return correlation

def extract_glcm_features(img):
    '''
    Extracts the entropy, homogeneity, and correlation features
    from the GLCM of an image.
    '''
    glcm_matrix = glcm(img)
    print(glcm_matrix.shape)
    entropy = extract_entropy(img, glcm_matrix)
    homogeneity = extract_homogeneity(img, glcm_matrix)
    correlation = extract_correlation(img, glcm_matrix)
    return entropy, homogeneity, correlation

def lbp(img):
    '''
    Calculates the Local Binary Pattern (LBP) of an image. 
    This works 
    '''
    lbp = local_binary_pattern(img, 8, 1, 'uniform')
    return lbp
    

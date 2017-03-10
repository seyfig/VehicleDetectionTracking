import numpy as np
import cv2
import matplotlib.image as mpimg

import os

import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import KFold

# **KFOLD TEST FOR ...**

carfld = './../data/vehicles/'
nonfld = './../data/non-vehicles/'


def convert_color(img, frm="RGB", to="BGR"):
    converted = None
    if (frm == to):
        converted = np.copy(img)
    elif (frm == "RGB" and to == "BGR") or (frm == "BGR" and to == "RGB"):
        r, g, b = cv2.split(img)
        converted = cv2.merge((b, g, r))
    elif (frm == "RGB"):
        if to == 'HSV':
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif to == 'LUV':
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif to == 'HLS':
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif to == 'YUV':
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif to == 'YCrCb':
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            print('convert color error', frm, to)
    elif (frm == "BGR"):
        if to == 'HSV':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif to == 'LUV':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif to == 'HLS':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif to == 'YUV':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif to == 'YCrCb':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            print('convert color error', frm, to)
    else:
        print('convert color error', frm, to)
    return converted


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block,
                                                   cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis,
                       feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_hog(imgs, cspace='RGB', orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, "BGR", cspace)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(
                    get_hog_features(
                        feature_image[:, :, channel],
                        orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient,
                                            pix_per_cell, cell_per_block,
                                            vis=False,
                                            feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', color_space_spatial=True,
                     color_space_hist=True, orient=9,
                     pix_per_cell=8, cell_per_block=2,
                     hog_channel=0, spatial=32,
                     hist_bins=32, spatial_feat=True,
                     hist_feat=True, hog_feat=True, imreader='cv2'):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        img_features = []
        # Read in each one by one
        if imreader == 'cv2':
            # Read in each one by one
            image = cv2.imread(file)
            feature_image = convert_color(image, "BGR", color_space)
        else:
            image = mpimg.imread(file)
            feature_image = convert_color(image, "RGB", color_space)
            # apply color conversion if other than 'RGB'
        # Compute spatial features if flag is set
        if spatial_feat:
            spatial_size = (spatial, spatial)
            if color_space_spatial:
                spatial_features = bin_spatial(feature_image,
                                               size=spatial_size)
            else:
                spatial_features = bin_spatial(image,
                                               size=spatial_size)
            img_features.append(spatial_features)
        # Compute histogram features if flag is set
        if hist_feat:
            if color_space_hist:
                hist_features = color_hist(feature_image, nbins=hist_bins)
            else:
                hist_features = color_hist(image, nbins=hist_bins)
            img_features.append(hist_features)

        # 7) Compute HOG features if flag is set
        if hog_feat:
            hog_features = []
            for channel in range(feature_image.shape[2]):
                if channel in hog_channel:
                    hog_features.append(get_hog_features(
                                        feature_image[:, :, channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)

            # Append the new feature vector to the features list
            img_features.append(hog_features)
        features.append(np.concatenate(img_features))

    # Return list of feature vectors
    return features


num_samples = 50
cars = []
notcars = []
i = 0
for folder in os.walk(carfld):
    folder_path = folder[0]
    for image in folder[2]:
        if '.DS_Store' in image:
            continue
        i += 1
        if i > num_samples:
            break
        cars.append(folder_path + '/' + image)

i = 0
for folder in os.walk(nonfld):
    folder_path = folder[0]
    for image in folder[2]:
        if '.DS_Store' in image:
            continue
        i += 1
        if i > num_samples:
            break
        notcars.append(folder_path + '/' + image)


print('cars, notcars: ', len(cars), len(notcars))
# Split up data into randomized training and test sets

nfold = 2
nnfold = 5

allcars = cars + notcars
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
indices = np.arange(0, len(y))

kfolds = []
for i in range(nnfold):
    rand_state = np.random.randint(0, 100)
    kfold = KFold(n_splits=nfold, random_state=rand_state, shuffle=True)
    kfolds.append(kfold)
    for train, test in kfold.split(indices):
        print(i, len(train), sum(train),
              round(float(sum(train)) / len(train), 2),
              len(test), sum(test), round(float(sum(test)) / len(test), 2))

# TODO: Tweak these parameters and see how the results change.
imreader_list = ['cv2']
colorspace_list = ['YCrCb', 'YUV']
orient_list = [9, 6, 12, 8, 10, 7, 11]
pix_per_cell_list = [8, 16]
cell_per_block_list = [2,1]
hog_channel_list = [[0,1],[0,2],[1,2],[0,1,2],[0],[1],[2]]
spatial_list = [12,14,16,20,24,32]
histbin_list = [128]

color_space_spatial = True
color_space_hist = True
feat_list = ['spatial']

flog = open('log.csv', 'w')
flog.write('k,imreader,colorspace,orient,pix_per_cell,cell_per_block,hog_channel,spatial,histbin,feat,featvlen,convsec,trainsec,predsec,acc\n')
flog.close()

tall=time.time()
for i, colorspace in enumerate(colorspace_list):
    for j, orient in enumerate(orient_list):
        for c, cell_per_block in enumerate(cell_per_block_list):
            for h, hog_channel in enumerate(hog_channel_list):
                for p, pix_per_cell in enumerate(pix_per_cell_list):
                    for rd, imreader in enumerate(imreader_list):

                        for sp, spatial in enumerate(spatial_list):
                            for hb, histbin in enumerate(histbin_list):
                                for ft, feat in enumerate(feat_list):
                                    print('hog %s(%d)' % (hog_channel, h),
                                          'ppc %d(%d)' % (pix_per_cell, p),
                                          'cpb %d(%d)' % (cell_per_block, c),
                                          'csp %s(%d)' % (colorspace, i),
                                          'ori %d(%d)' % (orient, j),
                                          'imr %s(%d)' % (imreader, rd),
                                          'tot time:', round((time.time() - tall),3))
                                    spatial_feat = True
                                    hist_feat = True
                                    if feat == 'spatial':
                                        hist_feat = False
                                    elif feat == 'hist':
                                        spatial_feat = False
                                    t = time.time()
                                    allcars_features = extract_features(allcars,
                                                                        color_space=colorspace,
                                                                        color_space_spatial=color_space_spatial,
                                                                        color_space_hist=color_space_hist,
                                                                        orient=orient,
                                                                        pix_per_cell=pix_per_cell,
                                                                        cell_per_block=cell_per_block,
                                                                        hog_channel=hog_channel,
                                                                        spatial=spatial,
                                                                        hist_bins=histbin,
                                                                        spatial_feat=spatial_feat,
                                                                        hist_feat=hist_feat,
                                                                        hog_feat=True,
                                                                        imreader=imreader)
                                    convsec = time.time() - t

                                    X = np.asarray(allcars_features, dtype=np.float64)

                                    # Fit a per-column scaler
                                    X_scaler = StandardScaler().fit(X)

                                    # Apply the scaler to X
                                    scaled_X = X_scaler.transform(X)

                                    featvlen = len(scaled_X[0])

                                    k = 0

                                    for kfold in kfolds:
                                        for train, test in kfold.split(indices):
                                            X_train = scaled_X[train]
                                            y_train = y[train]
                                            X_test = scaled_X[test]
                                            y_test = y[test]
                                            # Use a linear SVC
                                            svc = LinearSVC()
                                            # Check the training time for the SVC
                                            t=time.time()
                                            svc.fit(X_train, y_train)
                                            t2 = time.time()
                                            score = svc.score(X_test, y_test)
                                            ttrain = t2-t

                                            t=time.time()
                                            n_prediction = svc.predict(X_test)
                                            t2 = time.time()
                                            tpred = t2 - t

                                            hog_str = str(hog_channel).replace(',','-')

                                            flog = open('log.csv', 'a')
                                            flog.write('%d,%s,%s,%d,%d,%d,%s,%d,%d,%s,%d,%f,%f,%f,%f\n' % (
                                                k,imreader,
                                                colorspace,orient,pix_per_cell,cell_per_block,hog_str,
                                                spatial,histbin,feat, featvlen,
                                                convsec,ttrain,tpred,score))
                                            flog.close()
                                            print(k,imreader,
                                                colorspace,orient,pix_per_cell,cell_per_block,hog_str,
                                                spatial,histbin,feat,featvlen,
                                                round(convsec,5),round(ttrain, 5),round(tpred, 5),round(score, 5))

                                            k += 1


print('tot time:', round((time.time() - tall),3))


for train, test in kfold.split(indices):
    print('sum trn' , sum(train), 'sum test' , sum(test))

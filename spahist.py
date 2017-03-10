import matplotlib.image as mpimg
import numpy as np
import cv2
import time
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(
        img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(
        img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(
        img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(
            feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


carfld = './../subset/vehicles_smallset/'
nonfld = './../subset/non-vehicles_smallset/'
# Read in car and non-car images
cars = []
notcars = []

for folder in os.walk(carfld):
    folder_path = folder[0]
    for image in folder[2]:
        if '.DS_Store' in image:
            continue
        if 'image' in image or 'extra' in image:
            notcars.append(folder_path + '/' + image)
        else:
            cars.append(folder_path + '/' + image)

for folder in os.walk(nonfld):
    folder_path = folder[0]
    for image in folder[2]:
        if '.DS_Store' in image:
            continue
        if 'image' in image or 'extra' in image:
            notcars.append(folder_path + '/' + image)
        else:
            cars.append(folder_path + '/' + image)

print('#cars:', len(cars), '#notcars:', len(notcars))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)

nfold = 5

allcars = cars + notcars
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
indices = np.arange(0, len(y))

kfold = KFold(n_splits=nfold, random_state=rand_state, shuffle=True)

for train, test in kfold.split(indices):
    print('sum trn', sum(train), 'sum test', sum(test))

# TODO play with these values to see how your classifier
# performs under different binning scenarios
# spatial_list = [8, 16, 32, 64, 128]
# histbin_list = [8, 16, 32, 64, 128]

# spatial_list = [8, 12, 16, 24, 32]
# histbin_list = [32, 48, 64, 96, 128]

# spatial_list = [8, 10, 12, 14, 16]
# histbin_list = [64, 96, 128, 192, 256]

# spatial_list = [12, 13, 14, 15, 16]
# histbin_list = [64, 80, 96, 112 , 128]

spatial_list = [11, 12, 13, 14, 15, 16]
histbin_list = [100, 114, 128, 142, 156]

result_list = np.zeros((len(spatial_list), len(histbin_list), nfold))

print('k', 'spatial', 'histbin', 'sec', 'acc')
tall = time.time()
for i, spatial in enumerate(spatial_list):
    for j, histbin in enumerate(histbin_list):

        allcars_features = extract_features(allcars,
                                            cspace='RGB',
                                            spatial_size=(spatial, spatial),
                                            hist_bins=histbin,
                                            hist_range=(0, 256))
        X = np.asarray(allcars_features, dtype=np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        k = 0
        for train, test in kfold.split(indices):
            X_train = scaled_X[train]
            y_train = y[train]
            X_test = scaled_X[test]
            y_test = y[test]

            svc = LinearSVC()
            t = time.time()
            svc.fit(X_train, y_train)
            t2 = time.time()
            score = svc.score(X_test, y_test)
            print(k, spatial, histbin, round(t2 - t, 2), round(score, 5))
            result_list[i, j, k] = score

            k += 1
print('tot time:', round((time.time() - tall), 3))

for train, test in kfold.split(indices):
    print('len trn', len(train), 'sum trn', sum(train),
          'len test', len(test), 'sum test', sum(test))

arl = result_list.mean(axis=2)
print(arl)

indice = np.argsort(arl, axis=None)[::-1]
r, c = divmod(indice, arl.shape[1])
for i, j in zip(r, c):
    print(spatial_list[i], histbin_list[j], arl[i, j])

sarl = arl.mean(axis=1)
sarli = sarl.argsort()[::-1]
for i in sarli:
    print('s', spatial_list[i], sarl[i])
harl = arl.mean(axis=0)
harli = harl.argsort()[::-1]
for i in harli:
    print('h', histbin_list[i], harl[i])

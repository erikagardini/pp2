from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as plt
from scipy.spatial import distance
import os
import cv2
import numpy as np
from skimage.measure import compare_ssim


def SSIM(img1, img2):
    (score, diff) = compare_ssim(img1, img2, full=True)
    return score


def sharpness(img1):
    gy, gx = np.gradient(img1)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = np.average(gnorm)
    return sharpness


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


DIR = "comparison/"
if not os.path.exists(DIR):
    os.mkdir(DIR)

#Number of waypoints
NC=20

#Load the dataset
x_test, y_test = loadlocal_mnist(
            images_path='datasets/t10k-images-idx3-ubyte',
            labels_path='datasets/t10k-labels-idx1-ubyte')

#Normalization and reshape
print("Data loaded \n\n")
x_test = x_test.astype('float64') / 255.
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#Some boundaries selected by visual inspection
boundaries = [[568,270], [21,313], [75,19], [307,169], [457,422], [446,105]]

for i in range(len(boundaries)):

    dir_res = DIR + "comparison_mnist_" + str(i)
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    print("Principal path from " + str(boundaries[i][0]) + " to " + str(boundaries[i][1]))

    X = x_test
    d = X.shape[1]

    boundary_ids = boundaries[i]

    #Plot the boundaries
    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[0]].reshape(28, 28))
    plt.savefig(dir_res + "/mnist_start_" + str(i) + ".png")
    plt.gray()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[1]].reshape(28, 28))
    plt.savefig(dir_res + "/mnist_end_" + str(i) + ".png")
    plt.gray()
    plt.close()

    s_span = np.array([10000, 1000, 100, 10, 0])
    ####PP NEW
    models_1 = np.load("new_principalpath/examples/mnist_" + str(i) + "/pp_models_" + str(i) + ".npy")
    ####PP_OLD
    models_3 = np.load(
        "original_principalpath/examples/mnist_" + str(i) + "_prefiltering_True/pp_models_" + str(i) + ".npy")

    for j, s in enumerate(s_span):
        path_1 = models_1[j, :, :]
        path_3 = models_3[j, :, :]

        dst_mat_1 = distance.cdist(path_1, X, 'euclidean')
        dst_mat_3 = distance.cdist(path_3, X, 'euclidean')

        min_distances_1 = np.min(dst_mat_1, axis=1)
        min_distances_3 = np.min(dst_mat_3, axis=1)

        plt.figure(figsize=(10, 8))
        plt.plot(min_distances_1, label="Proposed PP")
        plt.plot(min_distances_3, label="Original PP")
        plt.legend(fontsize=20)
        plt.title("Distance to the nearest neighbour", fontsize=20)
        plt.xlabel("Waypoints", fontsize=20)
        plt.ylabel("Distance", fontsize=20)
        plt.savefig(dir_res + "/" + str(i) + "_" + str(j))
        plt.close()

        nn_figs_1 = X[np.argmin(dst_mat_1, axis=1)]
        nn_figs_3 = X[np.argmin(dst_mat_3, axis=1)]

        SSIM_1 = []
        SSIM_3 = []
        sharpness_1 = []
        sharpness_3 = []

        blur_1 = []
        blur_3 = []

        for z in range(path_1.shape[0]):
            value = SSIM(path_1[z, :], nn_figs_1[z, :])
            SSIM_1.append(value)
            value = sharpness(path_1[z, :].reshape(28, 28))
            sharpness_1.append(value)
            value = variance_of_laplacian(path_1[z, :].reshape(28, 28))
            blur_1.append(value)

            value = SSIM(path_3[z, :], nn_figs_3[z, :])
            SSIM_3.append(value)
            value = sharpness(path_3[z, :].reshape(28, 28))
            sharpness_3.append(value)
            value = variance_of_laplacian(path_3[z, :].reshape(28, 28))
            blur_3.append(value)

        plt.figure(figsize=(10, 8))
        plt.plot(SSIM_1, label="Proposed PP")
        plt.plot(SSIM_3, label="Original PP")
        plt.legend(fontsize=20)
        plt.title("Structural Similarity Index (SSIM)", fontsize=20)
        plt.xlabel("Waypoints", fontsize=20)
        plt.ylabel("SSIM value", fontsize=20)
        plt.savefig(dir_res + "/" + str(i) + "_" + str(j) + "SSIM")
        plt.close()

        sharpness_1 = np.array(sharpness_1)
        sharpness_3 = np.array(sharpness_3)

        plt.figure(figsize=(10, 8))
        plt.plot(sharpness_1, label="Proposed PP")
        plt.plot(sharpness_3, label="Original PP")
        plt.legend(fontsize=20)
        plt.title("Sharpness", fontsize=20)
        plt.xlabel("Waypoints", fontsize=20)
        plt.ylabel("Average gradient magnitude", fontsize=20)
        plt.savefig(dir_res + "/" + str(i) + "_" + str(j) + "sharpness")
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(blur_1, label="Proposed PP")
        plt.plot(blur_3, label="Original PP")
        plt.legend(fontsize=20)
        plt.title("Blur detection", fontsize=20)
        plt.xlabel("Waypoints", fontsize=20)
        plt.ylabel("Laplacian variance", fontsize=20)
        plt.savefig(dir_res + "/" + str(i) + "_" + str(j) + "blur")
        plt.close()

    print("\n\n\n")
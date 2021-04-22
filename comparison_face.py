from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
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
data = fetch_olivetti_faces()
targets = data.target

#Normalization and reshape
print("Data loaded \n\n")
data = data.images.reshape((len(data.images), -1))
data = data.astype('float64') / 255.

#Some boundaries selected by visual inspection
boundaries = [[0,399], [10,39], [40,299]]

for i in range(len(boundaries)):
    #Create the folder to save the results
    dir_res = DIR + "comparison_face_" + str(i)
    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    print("Principal path from " + str(boundaries[i][0]) + " to " + str(boundaries[i][1]))

    X = data
    d = X.shape[1]

    boundary_ids = boundaries[i]

    #Plot the boundaries
    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[0]].reshape(64, 64))
    plt.gray()
    plt.savefig(dir_res + "/start_" + str(i) + ".png")
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(X[boundary_ids[1]].reshape(64, 64))
    plt.gray()
    plt.savefig(dir_res + "/end_" + str(i) + ".png")
    plt.close()

    s_span = np.array([10000, 1000, 100, 10, 0])
    ####PP NEW
    models_1 = np.load("new_principalpath/examples/face_"+str(i)+"/pp_models_"+ str(i)+".npy")
    ####PP_OLD
    models_3 = np.load("original_principalpath/examples/face_"+str(i)+"_prefiltering_True/pp_models_"+ str(i)+".npy")

    #Plot the nearest picture corresponding to the waypoints of each model
    for j, s in enumerate(s_span):
        path_1 = models_1[j, :, :]
        path_3 = models_3[j, :, :]

        dst_mat_1 = distance.cdist(path_1, X, 'euclidean')
        dst_mat_3 = distance.cdist(path_3, X, 'euclidean')

        min_distances_1 = np.min(dst_mat_1, axis=1)
        min_distances_3 = np.min(dst_mat_3, axis=1)

        plt.figure(figsize=(10, 8))
        plt.plot(min_distances_1, label="Revised PP", linewidth=5)
        plt.plot(min_distances_3, label="Original PP", linewidth=5)
        plt.legend(fontsize=30)
        plt.title("Distance to the nearest neighbour", fontsize=30)
        plt.xlabel("Waypoints", fontsize=30)
        plt.ylabel("Distance", fontsize=30)
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
            value = sharpness(path_1[z, :].reshape(64, 64))
            sharpness_1.append(value)
            value = variance_of_laplacian(path_1[z, :].reshape(64, 64))
            blur_1.append(value)

            value = SSIM(path_3[z, :], nn_figs_3[z, :])
            SSIM_3.append(value)
            value = sharpness(path_3[z, :].reshape(64, 64))
            sharpness_3.append(value)
            value = variance_of_laplacian(path_3[z, :].reshape(64, 64))
            blur_3.append(value)

        plt.figure(figsize=(10, 8))
        plt.plot(SSIM_1, label="Revised PP", linewidth=5)
        plt.plot(SSIM_3, label="Original PP", linewidth=5)
        plt.legend(fontsize=30)
        plt.title("SSIM", fontsize=30)
        plt.xlabel("Waypoints", fontsize=30)
        plt.ylabel("SSIM value", fontsize=30)
        plt.savefig(dir_res + "/" + str(i) + "_" + str(j) + "SSIM")
        plt.close()

        sharpness_1 = np.array(sharpness_1)
        sharpness_3 = np.array(sharpness_3)

        plt.figure(figsize=(10, 8))
        plt.plot(sharpness_1, label="Revised PP", linewidth=5)
        plt.plot(sharpness_3, label="Original PP", linewidth=5)
        plt.legend(fontsize=30)
        plt.title("Sharpness", fontsize=30)
        plt.xlabel("Waypoints", fontsize=30)
        plt.ylabel("Average gradient magnitude", fontsize=30)
        plt.savefig(dir_res + "/" + str(i) + "_" + str(j) + "sharpness")
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(blur_1, label="Revised PP", linewidth=5)
        plt.plot(blur_3, label="Original PP", linewidth=5)
        plt.legend(fontsize=30)
        plt.title("Blur detection", fontsize=30)
        plt.xlabel("Waypoints", fontsize=30)
        plt.ylabel("Laplacian variance", fontsize=30)
        plt.savefig(dir_res + "/" + str(i) + "_" + str(j) + "blur")
        plt.close()

    print("\n\n\n")
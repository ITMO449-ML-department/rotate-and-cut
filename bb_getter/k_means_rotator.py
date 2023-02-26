import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import PIL


# TODO: FIX THIS SHIT
def _k_means_choose_channel(new_X, image, verbose,save_path):
    first_channel = (new_X*(new_X == 1)).reshape((*image.shape[:-1], 1))
    second_channel = (new_X*(new_X == 2)).reshape((*image.shape[:-1], 1))
    third_channel = (new_X*(new_X == 3)).reshape((*image.shape[:-1], 1))
    
    if save_path is not None:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(first_channel,cmap="gray")
        ax[1].imshow(second_channel,cmap="gray")
        ax[2].imshow(third_channel,cmap="gray")
        fig.savefig(save_path+"kmeans.jpg")
        fig.clear()

    contours_counter = []
    for img in [first_channel, second_channel, third_channel]:
        img = np.array([img, img, img]).reshape((image.shape))
        img_grey = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        thresh = 0
        ret, thresh_img = cv2.threshold(img_grey.astype(
            np.uint8), thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_counter.append(len(contours))
        if verbose == 2:
            print(f"Amount {len(contours_counter)} -> {contours_counter[-1]}")
    goal_index = np.argmax(np.array(contours_counter)) + 1
    return goal_index


def _get_kmeans_mask(image, verbose,save_path=None):
    predictions = KMeans(3, n_init=10).fit_predict(image.reshape((-1, 3))) + 1
    goal_index = _k_means_choose_channel(predictions, image, verbose,save_path)
    lines_mask = (predictions*(predictions == goal_index)).reshape((*image.shape[:-1], 1))
    return lines_mask


def _get_hough(lines_mask, image):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    return cv2.HoughLinesP(lines_mask.astype("uint8"), rho, theta, threshold, np.array([]).astype("uint8"),
                            min_line_length, max_line_gap), line_image


def _get_hough_angles(lines_mask, image, save_path):
    lines, line_image = _get_hough(lines_mask, image)

    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angles.append(np.arctan2(y2-y1, x2-x1))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    angles = np.array(angles)
    
    if save_path is not None:
        plt.imsave(save_path+"hough.jpg",line_image)
    return angles


def _get_rotation_angle_with_dbscan(angles):
    angles = (angles * 180 / np.pi).reshape((-1, 1))

    db = DBSCAN(eps=1, min_samples=100).fit(angles)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    card = []
    for i in range(n_clusters_):
        card.append((labels == i).sum())

    rot_angle = angles[labels == card.index(max(card))].mean()
    return rot_angle


def get_rotation_angle(name, verbose, save_path = None):
    """
    
    :name: path to image
    :verbose:  0 - no info; 1 - results, important info; 2 - every step

    :return: angle to rotate, mask made by k-means algo

    """


    if verbose == 2:
        print(f"Openning image {name}")
    image = cv2.imread(name)
    if verbose == 2:
        print("Getting mask of rows using k-means algo")
    lines_mask = _get_kmeans_mask(image, verbose=verbose,save_path=save_path)

    if verbose == 2:
        print("Calculating hough lines and angles")
    hough_angles = _get_hough_angles(lines_mask, image, save_path)

    return _get_rotation_angle_with_dbscan(hough_angles), lines_mask

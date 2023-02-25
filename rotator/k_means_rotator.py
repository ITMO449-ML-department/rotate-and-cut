import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import PIL

def _get_kmeans_mask(image, verbose):
    X = image.reshape((-1,3))
    km = KMeans(3, n_init=10)
    new_X = km.fit_predict(X)
    new_X += 1

    card = []
    for i in range(3):
        lines_mask = (new_X*(new_X == i+1)).reshape(image.shape[:-1])
        card.append((new_X == i+1).sum())

    first_channel = (new_X*(new_X == 1)).reshape((*image.shape[:-1], 1)) 
    second_channel = (new_X*(new_X == 2)).reshape((*image.shape[:-1], 1)) 
    third_channel = (new_X*(new_X == 3)).reshape((*image.shape[:-1], 1))

    contours_counter = []
    max_s = 0
    max_s_indx = 0
    s_arr = []
    for img in [first_channel, second_channel, third_channel]:
        s = 0
        img = np.array([img,img,img])
        img = img.reshape((image.shape))
        #convert img to grey
        img_grey = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # img_grey = img
        #set a thresh
        thresh = 0
        #get threshold image
        ret,thresh_img = cv2.threshold(img_grey.astype(np.uint8), thresh, 255, cv2.THRESH_BINARY)
        #find contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_counter.append(len(contours))
        for ctr in contours:
            s += cv2.contourArea(ctr)
        # print(s)
        s_arr.append(s)
        if verbose == 2:
            print(f"Amount {len(contours_counter)} -> {contours_counter[-1]}")
    # goal_sort = np.argsort(np.array(contours_counter)) 
    # print(contours_counter)
    # print(s_arr)
    # print(goal_sort)
    # if contours_counter[goal_sort[1]]/ contours_counter[goal_sort[2]] > 0.75:
    #     print("S close")
    #     if s_arr[goal_sort[2]] > s_arr[goal_sort[1]]:
    #         goal_index = 3
    #     else:
    #         goal_index = 2
    # else:
    #     goal_index = goal_sort[2] + 1
    goal_index = np.argmax(np.array(contours_counter)) + 1
    if verbose == 2:
        print(f"Rows class -> {goal_index}")

    lines_mask = (new_X*(new_X == goal_index)).reshape((*image.shape[:-1], 1)) 
    return lines_mask

def _get_hough_angles(lines_mask, image, plot_hough):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    angles = []

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(lines_mask.astype("uint8"), rho, theta, threshold, np.array([]).astype("uint8"),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            angles.append(np.arctan2(y2-y1, x2-x1))
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    angles = np.array(angles)
    if plot_hough:
        plt.imshow(line_image)
    return angles

def _get_rotation_angle_with_dbscan(angles):
    angles = angles* 180 / np.pi
    angles= angles.reshape((-1,1))
    db = DBSCAN(eps=1, min_samples=100).fit(angles)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    card = []
    for i in range(n_clusters_):
        card.append((labels == i).sum())
    rot_angle = angles[labels == card.index(max(card))].mean()
    return rot_angle


def get_rotation_angle(name, verbose, plot_kmeans = False, plot_hough=False):
    if verbose == 2:
        print(f"Openning image {name}")
    image = cv2.imread(name)
    if verbose == 2:
        print("Getting mask of rows using k-means algo")
    lines_mask = _get_kmeans_mask(image, verbose=verbose)
    if verbose == 2:
        print("Calculating hough lines and angles")
    hough_angles = _get_hough_angles(lines_mask, image, plot_hough)
    return _get_rotation_angle_with_dbscan(hough_angles), lines_mask

import numpy as np
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps
import bb_getter.k_means_rotator as k_means_rotator
import os
import scipy


def _rotate_bb(image_original, image_original_rotated_array, angle, bboxes, save_path):
    new_rect_rotated = np.zeros_like(image_original)
    colors = [(0,255,0), (255, 0, 0), (0,0,255)]
    i = 0
    rotated = []
    for bbox in bboxes:
        p1,p2,p3,p4 = bbox
        y1, x1 = p1
        y2, x2 = p2
        y3, x3 = p3
        y4, x4 = p4
        center = np.array([image_original_rotated_array.shape[1]//2,image_original_rotated_array.shape[0]//2]).astype(int)
        rotate_matrix = cv2.getRotationMatrix2D(center=center.tolist(), angle=angle, scale=1)
        rotate_matrix = rotate_matrix[:,:-1]
        p1 = np.array([x1 - center[0],y1 - center[1]]).dot(rotate_matrix).astype(int)
        p2 = np.array([x2 - center[0],y2 - center[1]]).dot(rotate_matrix).astype(int)
        p3 = np.array([x3 - center[0],y3 - center[1]]).dot(rotate_matrix).astype(int)
        p4 = np.array([x4 - center[0],y4 - center[1]]).dot(rotate_matrix).astype(int)
        center = np.array([new_rect_rotated.shape[1]//2,new_rect_rotated.shape[0]//2]).astype(int)
        new_rect_rotated = cv2.circle(new_rect_rotated, p1 + center,2, colors[i], 30)
        new_rect_rotated = cv2.circle(new_rect_rotated, p2 + center,2, colors[i], 30)
        new_rect_rotated = cv2.circle(new_rect_rotated, p3 + center,2, colors[i], 30)
        new_rect_rotated = cv2.circle(new_rect_rotated, p4 + center,2, colors[i], 30)
        rotated.append([p1+center, p2+center, p3+center, p4+center,])
        i = (i+1)%3
    final  = cv2.addWeighted(image_original,1,new_rect_rotated,1,0)
    if save_path is not None:
        plt.imsave(save_path + "bboxes.jpg", final)
    return rotated


def smooth(y, x):
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(len(x), x[1]-x[0])
    spectrum = w**2

    cutoff_idx = spectrum < (spectrum.max()/100)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)
    return y2

def _calculate_intensities(rows_image_rotated_array, image_original_rotated_array, verbose):
    if verbose == 2:
        print("Calculating intensities")
    parts = rows_image_rotated_array.shape[0] // 6
    # parts = max(800, rows_image_rotated_array.shape[0] // 6)
    # print(parts)
    # parts = 800
    mn = []
    inds = np.linspace(0, rows_image_rotated_array.shape[0]-1, parts+1).astype(int)
    for i in range(1, parts+1):
        if np.count_nonzero(image_original_rotated_array[inds[i-1]:inds[i], :, 1]) > 0:
            mn.append(np.count_nonzero(rows_image_rotated_array[inds[i-1]:inds[i], :, 1])/np.count_nonzero(image_original_rotated_array[inds[i-1]:inds[i], :, 1]))
        else:
            mn.append(0)
    mn_smoothed = smooth(mn,[i for i in range(len(mn))])
    limit_smoothed = max(mn_smoothed)*0.4
    return mn, parts, inds, limit_smoothed


def _calculate_limit(mn, save_path, limit_smoothed, verbose):
    # limit = max(sorted(mn)[:int(len(mn)*0.95)]) * 0.65
    limit = max(mn)*0.4
    print(limit)
    if save_path is not None:
        if verbose == 2:
            print("Plotting gists")
        plt.plot(mn)
        plt.plot([i for i in range(0, len(mn))], [limit] * len(mn), label='not')
        plt.plot([i for i in range(0, len(mn))], [limit_smoothed] * len(mn), label='smoothed')
        plt.legend()
        plt.savefig(save_path + "rows_gists.jpg")
        plt.close()
    return limit
    

def _prepare_save_location(save_path, name, verbose):
    if save_path is not None:
        if verbose == 2:
            print("Preparing save path")
        save_path = f"{save_path}/{name.split('/')[-1].split('.')[0]}/"
        if verbose > 0:
            print("Saving to", save_path)
        os.makedirs(save_path, exist_ok=True)
    return save_path


def _find_borders(parts, mn, inds, limit, verbose):
    if verbose == 2:
        print("Searching for rows")
    borders = []
    low = 0
    high = 0
    previous_is_row = False
    for i in range(1, parts+1):
        if mn[i-1] > limit:
            if not previous_is_row:
                low = inds[i-1]
                high = inds[i]
                previous_is_row = True
            else:
                high  = inds[i]
        else:
            if previous_is_row:
                borders.append((low, high))
            previous_is_row = False
    if verbose > 0:
        print(f"Found {len(borders)} rows")
    return borders


def _plot_orig_cut_kmeans(image_original_rotated_array, parts, mn, limit, inds, rows_image_rotated_array, save_path, verbose):
    if save_path is not None:
        if verbose == 2:
            print("Plotting comparison")
        image_array_for_drawing = image_original_rotated_array.copy()
        for i in range(1, parts+1):
            if mn[i-1] < limit:
                image_array_for_drawing[inds[i-1]:inds[i], :, :] = np.zeros_like(image_array_for_drawing[inds[i-1]:inds[i], :, :])

        f, axs = plt.subplots(1, 3, figsize=(12,8))
        axs[0].imshow(image_original_rotated_array)
        axs[1].imshow(image_array_for_drawing)
        axs[2].imshow(rows_image_rotated_array)
        f.savefig(save_path + "rows_comparison.jpg")
        f.clear()

def _find_bboxes(borders, image_original_rotated_array, save_path, verbose):
    bboxes = []
    bboxes_debug = []
    if verbose == 2:
        print("Calculating bboxes")
    for low, high in borders:
        strip = image_original_rotated_array[low:high, :, :]
        counter_left = 0
        counter_right = strip.shape[1]-1
        # print(strip.shape)
        while True:
            if np.any(strip[:, counter_left, :] != 0):
                break
            else:
                # print(strip[:, counter_left, :])
                counter_left += 1
                if counter_left == strip.shape[1]-1:
                    # print("limit")
                    break
        while True:
            if np.any(strip[:, counter_right, :] != 0):
                break
            else:
                counter_right-= 1
                if counter_right < 1:
                    # print("limit")
                    break
        bboxes_debug.append(((low, high, counter_left, counter_right)))
        bboxes.append(((low, counter_left), (low, counter_right), (high, counter_right), (high, counter_left)))

    if verbose > 0:    
        print(f"Found {len(bboxes)} bboxes")
        
    if save_path is not None:
        image_array_for_drawing = image_original_rotated_array.copy() * 0 + 255
        # if verbose == 2:
        #     print("Plotting rows debug")
        # for low, high, left, right in bboxes_debug:
        #     image_array_for_drawing[low:high, left:right, :] = image_original_rotated_array[low:high, left:right, :]
        # # plt.imshow(image_array_for_drawing)
        # plt.imsave(save_path + "rows_debug.jpg", image_array_for_drawing)
        if verbose == 2:
            print("Plotting rows check")
        for box in bboxes:
            a, b = box[0]
            c, d = box[2]
            new_rect_rotated = cv2.rectangle(image_array_for_drawing.copy() * 0, (b, a), (d, c), (255,0,0), -1)
            image_original_rotated_array  = cv2.addWeighted(image_original_rotated_array,1,new_rect_rotated,1,0)
        plt.imsave(save_path + "rows_check.jpg", image_original_rotated_array)
    return bboxes


def get_bb(name, save_path=None, verbose = 0):
    """
    
    :name: path to image
    :save_path: save path for plots
    :verbose:  0 - no info; 1 - results, important info; 2 - every step

    :return: array of bounding boxes, angle of rotation

    """
    
    save_path = _prepare_save_location(save_path, name, verbose)

    angle, lines_mask = k_means_rotator.get_rotation_angle(name, verbose,save_path)

    if verbose == 2:
        print("Preparing arrays")
    image_original = Image.open(name)
    image_original_array = np.array(image_original)
    image_original_rotated = image_original.rotate(angle, expand=True)
    image_original_rotated_array = np.array(image_original_rotated)

    lines_mask_tiled = np.zeros_like(image_original_array)
    lines_mask_tiled[:,:,0] = lines_mask[:,:,0]
    lines_mask_tiled[:,:,1] = lines_mask[:,:,0]
    lines_mask_tiled[:,:,2] = lines_mask[:,:,0]
    lines_mask_tiled *= 255

    rows_image_array = lines_mask_tiled
    rows_image_rotated = Image.fromarray(lines_mask_tiled).rotate(angle, expand=True)
    rows_image_rotated_array = np.array(rows_image_rotated)
    
    if save_path is not None:
        plt.imsave(save_path + "rows_mask.jpg", lines_mask_tiled)

    mn, parts, inds, limit_smoothed = _calculate_intensities(rows_image_rotated_array, image_original_rotated_array, verbose)

    limit = _calculate_limit(mn, save_path, limit_smoothed, verbose)

    limit = limit_smoothed
    
    _plot_orig_cut_kmeans(image_original_rotated_array, parts, mn, limit, inds, rows_image_rotated_array, save_path, verbose)
    
    borders = _find_borders(parts, mn, inds, limit, verbose)

    bboxes = _find_bboxes(borders, image_original_rotated_array, save_path, verbose)

    return _rotate_bb(image_original_array, image_original_rotated_array, angle, bboxes, save_path)
                    
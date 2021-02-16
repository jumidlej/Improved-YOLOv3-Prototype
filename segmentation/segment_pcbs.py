import cv2
import numpy as np
import math
import sys
import imutils
from os import listdir

'''
Does: list .png files
Arguments: images path (.png)
Returns: list of images names (without .png)
'''
def list_png_files(path=None):
    if path == None:
        print("Nenhuma pasta foi especificada.")
        return 0

    images_names = []
    files = [f for f in listdir(path)]
    for f in files:
        if f[len(f)-4:] == ".png":
            images_names.append(f[:-4])

    return images_names

'''
Does: load an image
Arguments: image file
Returns: image
'''
def load_image(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    return image

'''
Does: thresholding
Arguments: image, dilatation
Returns: binary image
'''
def thresholding(image, dilate=0):
    # transforma imagem rgb em hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # binarização da imagem
    green_lower = (0, 0, 50)
    green_upper = (100, 100, 200)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # dilate
    mask = cv2.dilate(mask, None, iterations=dilate)

    return mask

'''
Does: find the two biggest contours (area)
Arguments: binary image, original image
Returns: contours, image with the contours
'''
def max_area_contour(mask, image):
    # contornos
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    pcbs_contour = []
    area = []

    for contour in contours:
        area.append(cv2.contourArea(contour))

    area = sorted(area)
    # print(area)
    
    contour_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    
    for contour in contours:
        if cv2.contourArea(contour) == area[-1]:
            pcbs_contour.append(contour)
            contour_image = cv2.drawContours(contour_image, [contour], -1, 255, 1)
        if cv2.contourArea(contour) == area[-2]:
            pcbs_contour.append(contour)
            contour_image = cv2.drawContours(contour_image, [contour], -1, 255, 1)

    return pcbs_contour, contour_image

'''
Does: two rectangles containing the contours
Arguments: contours, original image
Returns: rectangles, image with the rectangles
'''
def rotated_rectangle(contours, image):
    rects = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        # print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(rect)
        image = cv2.drawContours(image,[box],0,(0,0,255),2)

    return rects, image

'''
Does: crop a rectangle and turn it in to an image
Arguments: rectangle, original image
Returns: image cropped
'''
def crop_rotated_rectangle(rectangle, image, additional_width=0, additional_height=0):
    rect = ((rectangle[0][0], rectangle[0][1]), (rectangle[1][0] + additional_width, rectangle[1][1] + additional_height), rectangle[2])
    
    # rotate img
    angle = rect[2]
    rows,cols = image.shape[0], image.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(image,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    # print(box)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
    pts[1][0]:pts[2][0]]

    return img_crop

'''
Does: find the two pcbs and crop them in to two different images
Arguments: images path, cropped images path, image name
'''
def segment_pcbs(image_path, results_path):
    image_name = image_path.split("/")[-1]
    extension = image_name.split(".")[1]
    image_name = image_name.split(".")[0]
    # print(image_name)
    # print(extension)

    # carregar imagem
    image = load_image(image_path)
    cv2.imwrite(results_path+image_name+"_1."+extension, image)

    # binarização da imagem
    mask = thresholding(image, dilate=1)
    cv2.imwrite(results_path+image_name+"_2."+extension, mask)

    # maior contorno (placa)
    contours, image_contour = max_area_contour(mask, image.copy())
    cv2.imwrite(results_path+image_name+"_3."+extension, image_contour)

    # retângulo (placa)
    rectangles, image_rectangle = rotated_rectangle(contours, image.copy())
    cv2.imwrite(results_path+image_name+"_4."+extension, image_rectangle)
    
    cropped_image = crop_rotated_rectangle(rectangles[0], image.copy(), 120, 10)
    cv2.imwrite(results_path+image_name+"_5."+extension, cropped_image)

    cropped_image = crop_rotated_rectangle(rectangles[1], image.copy(), 120, 10)
    cv2.imwrite(results_path+image_name+"_6."+extension, cropped_image)


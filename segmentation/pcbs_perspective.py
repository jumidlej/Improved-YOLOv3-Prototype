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

def load_image(image_file):
    # ler imagem
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    return image

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
Does: find the biggest contours (area)
Arguments: binary image, original image
Returns: contours, binary image with the contour
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
    
    contour_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    
    for contour in contours:
        if cv2.contourArea(contour) == area[-1]:
            pcbs_contour.append(contour)
            contour_image = cv2.drawContours(contour_image, [contour], -1, 255, 1)

    return pcbs_contour, contour_image

def find_lines(image):
    left = image.copy()
    for i in range(0, left.shape[0]):
        point = True
        for j in range(0, left.shape[1]):
            if point and left[i][j] == 255:
                point = False
            else:
                left[i][j] = 0
    
    up = image.copy()
    for i in range(0, up.shape[1]):
        point = True
        for j in range(0, up.shape[0]):
            if point and up[j][i] == 255:
                point = False
            else:
                up[j][i] = 0
                
    right = image.copy()
    for i in reversed(range(right.shape[0])):
        point = True
        for j in reversed(range(right.shape[1])):
            if point and right[i][j] == 255:
                point = False
            else:
                right[i][j] = 0
    
    bottom = image.copy()
    for i in reversed(range(bottom.shape[1])):
        point = True
        for j in reversed(range(bottom.shape[0])):
            if point and bottom[j][i] == 255:
                point = False
            else:
                bottom[j][i] = 0
                
    return left, right, bottom, up

def hough_lines(image, line_size):
    lines = cv2.HoughLines(image,1,np.pi/180, line_size)
    if lines is not None:
        # pode ter mais de uma linha
        for rho,theta in lines[0]:
            r, t = rho, theta
    else:
        print("Não achou nenhuma linha")
    return r, t

def draw_line(rho, theta, img):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),255,1)
    return img

def interLines(rhoX, thetaX, rhoY, thetaY):
    a = np.array([[np.cos(thetaX), np.sin(thetaX)], [np.cos(thetaY), np.sin(thetaY)]])
    b = np.array([rhoX,rhoY])
    x = np.linalg.solve(a, b)
    return x[0],x[1]

def define_points(x, y, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x1 = int(x+30*(-b))
    y1 = int(y+30*(a))
    x2 = int(x-250*(-b))
    y2 = int(y-250*(a))
    return x1, y1, x2, y2

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

def pcb_final_cut(image_path, results_path):
    image_name = image_path.split("/")[-1]
    extension = image_name.split(".")[1]
    image_name = image_name.split(".")[0]
    # print(image_name)
    # print(extension)

    image = load_image(image_path)

    mask = thresholding(image.copy(), dilate=1)
    cv2.imwrite(results_path+image_name+"_1."+extension, mask)

    contours, image_contour = max_area_contour(mask, image.copy())
    cv2.imwrite(results_path+image_name+"_2."+extension, image_contour)

    left, right, bottom, up = find_lines(image_contour)
    cv2.imwrite(results_path+image_name+"_3."+extension, bottom)
    cv2.imwrite(results_path+image_name+"_4."+extension, up)
    cv2.imwrite(results_path+image_name+"_5."+extension, left)

    bottom_line = hough_lines(bottom, 30)
    bottom = draw_line(bottom_line[0], bottom_line[1], bottom)
    cv2.imwrite(results_path+image_name+"_6."+extension, bottom)

    up_line = hough_lines(up, 30)
    up = draw_line(up_line[0], up_line[1], up)
    cv2.imwrite(results_path+image_name+"_7."+extension, up)

    left_line = hough_lines(left, 30)
    left = draw_line(left_line[0],left_line[1], left)
    cv2.imwrite(results_path+image_name+"_8."+extension, left)

    angle = (bottom_line[1]+up_line[1])/2

    top_left_corner = interLines(up_line[0], up_line[1], left_line[0], left_line[1])
    bottom_left_corner = interLines(bottom_line[0], bottom_line[1], left_line[0], left_line[1])

    corners_image = cv2.circle(image.copy(), bottom_left_corner, 3, (0, 0, 255), 1)
    corners_image = cv2.circle(corners_image, top_left_corner, 3, (0, 0, 255), 1)
    cv2.imwrite(results_path+image_name+"_9."+extension, corners_image)

    x1, y1, x2, y2 = define_points(top_left_corner[0], top_left_corner[1], angle)
    x3, y3, x4, y4 = define_points(bottom_left_corner[0], bottom_left_corner[1], angle)

    corners_image = cv2.circle(corners_image, (x1, y1), 3, (0, 255, 255), 1)
    corners_image = cv2.circle(corners_image, (x2, y2), 3, (0, 255, 255), 1)
    corners_image = cv2.circle(corners_image, (x3, y3), 3, (0, 255, 255), 1)
    corners_image = cv2.circle(corners_image, (x4, y4), 3, (0, 255, 255), 1)
    cv2.imwrite(results_path+image_name+"_10."+extension, corners_image)

    angle = math.atan2(abs(y1-y2),abs(x1-x2))
    if (y1 > y2):
        angle *= -1
    angle = angle*(180/3.14159)
    center_x = int((x1 + x4) / 2)
    center_y = int((y1 + y4) / 2)
    width = (x1 - x2)**2 + (y1 - y2)**2
    width = width**0.5 
    height = (x1 - x3)**2 + (y1 - y3 )**2
    height = height**0.5

    rectangle = ((center_x, center_y), (width,height), angle)

    pcb = crop_rotated_rectangle(rectangle, image.copy())
    cv2.imwrite(results_path+image_name+"_11."+extension, pcb)



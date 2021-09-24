import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-in", "--input", required = True, help = "input image path")
args = vars(ap.parse_args())

# function for calculate canny with different sigmas
def auto_canny(image, sigma = 0.33):
    v = np.median(image)
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    edge = cv2.Canny(image, lower, upper)
    return edge

# image_path
image_path = args["input"]

# read gray image 
gray_img = cv2.imread(image_path,0)
# read color image
color_img = cv2.imread(image_path)
# resize read images
gray_img = cv2.resize(gray_img, (640,480))
color_img = cv2.resize(color_img, (640,480))

# apply threshold
thr = cv2.threshold(gray_img, 127,255,cv2.THRESH_BINARY)[1]

# apply findContours to get boundary points
contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# calculate areas of boundary points
areas = [cv2.contourArea(c) for c in contours]

if len(areas) != 0:
    # index of the largest area
    max_id = np.argmax(areas)
    
    # the points that make up the largest area
    cnt = contours[max_id]  
    # coordinates of bounding box
    x,y,w,h= cv2.boundingRect(cnt)
    # draw a rectangle(bounding box)
    cv2.rectangle(color_img, (x,y),(x+w,y+h),(0,255,0),4)
    
    # Separate the resulting area from the original image 
    img = gray_img[y:y+h,x:x+w]
    
    # apply adaptiveThreshold
    adaptive_thr = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1 )
    # apply auto_canny function
    edge = auto_canny(adaptive_thr)
    # apply findContours to get boundary points
    ctrs = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # contours sorted by coordinates x 
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    # calculate area of new image
    img_area = img.shape[0]*img.shape[1]
    
    # convert color
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # draw a rectangle around each of the numbers 
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h
        roi_ratio = roi_area/img_area
        if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
            if ((h>1.2*w) and (3*w>=h)):
                cv2.rectangle(image,(x,y),( x + w, y + h ),(0,0,255),1)

cv2.imshow('original image', color_img)
cv2.imshow('threshold', thr)
cv2.imshow('adaptive_thr', adaptive_thr)
cv2.imshow('edge', edge)
cv2.imshow('final', image)

cv2.waitKey(0)
cv2.destroyAllWindows()




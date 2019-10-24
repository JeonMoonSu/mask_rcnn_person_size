import cv2
import numpy as np
import math


protoFile = "/home/default/Deep-Learning/mask_rcnn_size_detection/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "/home/default/Deep-Learning/mask_rcnn_size_detection/pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

ppm = 9.0/20.0
xpm = 800.0/640.0
ypm = 600.0/480.0

def GetPersonPoint(img):
    running = True
    frame = img
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

# Empty list to store the detected keypoints
    points = []
    
    for i in range(nPoints):
    # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W 
        y = (frameHeight * point[1]) / H 

        if prob > threshold : 
            points.append((int(x), int(y)))
        else :
            points.append(None)
            running = False
    
    shoulder_distance = 0.0
    waist_distance = 0.0

    print("points: ",points)

    if running:
        cv2.line(frame, points[5], points[2], (0,255,255), 2)
        cv2.line(frame, points[11], points[8], (0,255,255), 2)

        cv2.imshow('Output-Skeleton', frame)
        cv2.imwrite('Output-Skeleton.jpg', frame)

        shoulder_dx = points[5][0] - points[2][0]
        shoulder_dy = points[5][1] - points[2][1]
        shoulder_dx = shoulder_dx*xpm
        shoulder_dy = shoulder_dy*ypm

        shoulder_distance = math.sqrt((shoulder_dx * shoulder_dx) + (shoulder_dy * shoulder_dy)) 

        waist_dx = points[11][0] - points[8][0]
        waist_dy = points[11][1] - points[8][1]
        waist_dx = waist_dx*xpm
        waist_dy = waist_dy*ypm

        waist_distance = math.sqrt((waist_dx * waist_dx) + (waist_dy * waist_dy))

    return shoulder_distance*ppm*1.18,waist_distance*ppm*1.4



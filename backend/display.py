import cv2
import numpy as np

def DisplayImg(img, points):
    points = [(int(round(points[i].item())), int(round(points[i+1].item()))) for i in range(0, len(points), 2)]
    for (x, y) in points:
        cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        
    return img
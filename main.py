import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

doNotPrint = [6, 13, 18]

def contours(planeEdges, planes):
    for i in range(len(planes)):
        std = np.std(planes[i])
        planes[i] = cv2.bilateralFilter(planes[i], -1, (1.0 - 0.3) * std, 10)
        v = np.median(planes[i])
        sigma1 = 0.5
        sigma2 = 0.4
        lower = int(max(0, (1.0 - sigma1) * v))
        upper = int(min(255, (1.0 + sigma2) * v))
        planeEdges.append(cv2.Canny(planes[i], lower, upper))
    return planeEdges

path =  "images/samolot"
names = [path + str(x) + ".jpg" if x > 9 else path + "0" + str(x) + ".jpg" for x in range(21)]
planes = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in names]

planeEdges = []
planeEdges = contours(planeEdges, planes)

for i in range(len(planeEdges)):
    planeEdges[i] = cv2.morphologyEx(planeEdges[i], cv2.MORPH_CLOSE, np.ones((5, 5),np.uint8))
    sth1, contour, sth2 = cv2.findContours(planeEdges[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for j in range(len(contour)):
        cv2.drawContours(planeEdges[i], contour, j, (255, 255, 255), cv2.FILLED)

betterEdges = []
betterEdges = contours(betterEdges, planeEdges)

color = [cv2.imread(x) for x in names]
coords = [0, 0, 0]

for i in range(len(betterEdges)):
    betterEdges[i] = cv2.morphologyEx(betterEdges[i], cv2.MORPH_CLOSE, np.ones((5, 5),np.uint8))
    sth1, contour, sth2 = cv2.findContours(betterEdges[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for j in range(len(contour)):
        if(contour[j].size < 200):
            continue
        M = cv2.moments(contour[j])
        randomColor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255));
        cv2.drawContours(color[i], contour, j, randomColor, thickness = 3)
        cv2.circle(color[i], (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), 5, randomColor, -1)

plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
fig, axes = plt.subplots(6, 3, figsize = (30, 42))
number = 0
for i in range(len(axes)):
    for j in range(3):
        while(number in doNotPrint):
            number += 1
        axes[i][j].get_xaxis().set_visible(False)
        axes[i][j].get_yaxis().set_visible(False)
        axes[i][j].imshow(cv2.cvtColor(color[number], cv2.COLOR_BGR2RGB), cmap = 'gray')
        cv2.imwrite(str(number) + '.png', color[number])
        number += 1
fig.savefig('mozaika.pdf')



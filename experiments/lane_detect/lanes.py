import cv2
import numpy as np
from math import hypot, degrees, tan, radians
from lane import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cap = cv2.VideoCapture("dikcekim_uncut.mp4")
ret, frame = cap.read()
#height, width, ch = frame.shape
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('perspective.mp4',fourcc, 3, (width, height))

ps = []
for i in range(10):
	ret, frame = cap.read()


print "lines are being found..."
index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
		lanes, perms = find_lanes(frame)
		ps.append(perms)
		#imname = "lanes/img{:0>5d}.png".format(index)
		#cv2.imwrite(imname,frame)
		#cimname = "lanes/img{:0>5d}.c.png".format(index)
		#cv2.imwrite(cimname,lanes)
		
		#out.write(lanes)
		index+=1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
    else:
        break
    if index > 10000:
    	break

# Release everything if job is finished
cap.release()
#out.release()
#cv2.destroyAllWindows()

data = adjust_pts(ps)

cap = cv2.VideoCapture("dikcekim_uncut.mp4")
ret, frame = cap.read()
height, width, ch = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('perspective.mp4',fourcc, 25, (width, height))

for i in range(10):
	ret, frame = cap.read()

print "lines are being found..."

index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
		thetas = []
		for i in range(4):
			theta = data[i][index][1]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = data[i][index][0]
			y0 = 0
			if x0==-1 and theta==-1:
				break

			thetas.append(theta)
				    
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			pts = np.array([[x0,y0]], np.int32)
			frame = cv2.polylines(frame,[pts],True,(0,0,255),2)

			cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
		
		rows, cols, ch = frame.shape
		degree = degrees(np.mean(thetas))
		M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
		dst = cv2.warpAffine(frame,M,(cols,rows))

		out.write(dst)
		index+=1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
    else:
        break
    if index > 10000:
    	break

# Release everything if job is finished
cap.release()

import cv2
import numpy as np
from math import hypot, degrees, tan, radians

cap = cv2.VideoCapture("edited.mp4")
ret, frame = cap.read()
height, width, ch = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('perspective2.mp4',fourcc, 20.0, (width, height))
index=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.bilateralFilter(gray, 11, 17, 17)
		edges = cv2.Canny(gray,100,200)
		lines = cv2.HoughLines(edges,1,np.pi/180,100)

		thetas = []
		pt_set = []

		for i in lines:
			pts = []
			for rho,theta in i:
			    a = np.cos(theta)
			    b = np.sin(theta)
			    x0 = a*rho
			    y0 = b*rho
			    
			    x1 = int(x0 + 100*(-b))
			    y1 = int(y0 + 100*(a))
			    x2 = int(x0 - 100*(-b))
			    y2 = int(y0 - 100*(a))

			    pt_set.append((x0,y0,rho))
			    pts = np.array([[x0,y0]], np.int32)
			    frame = cv2.polylines(frame,[pts],True,(0,0,255),2)

			    cv2.line(frame,(x1,y1),(x2,y2),(0,255,255),1)
			    
			    thetas.append(theta)
		"""
		thetas_d = []
		#for i in thetas:
		#	print degrees(i)
		for i in thetas:
			if degrees(i) > 90:
				thetas_d.append(degrees(i)-180)
			else:
				thetas_d.append(degrees(i))
		rows,cols,ch = frame.shape
		radian = np.mean(thetas)
		degree = np.mean(thetas_d)
		M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)

		dst = cv2.warpAffine(frame,M,(cols,rows))
		
		imname = "img{:0>5d}.png".format(index)
		c_imname = "img{:0>5d}.c.png".format(index)
		cv2.imwrite(c_imname,dst)
		cv2.imwrite(imname,frame)
		"""
		out.write(frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
    else:
        break
    index+=1
    print index

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


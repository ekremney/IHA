import cv2
import numpy as np
from math import hypot

cap = cv2.VideoCapture("edited.mp4")
ret, frame = cap.read()
height, width, ch = frame.shape
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
		
		frame = cv2.bilateralFilter(frame, 11, 17, 17)
		edges = cv2.Canny(frame,100,200)
		lines = cv2.HoughLines(edges,1,np.pi/180,90)
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
			    
			    pts = np.array([[x0,y0]], np.int32)
			    frame = cv2.polylines(frame,[pts],True,(0,0,255),2)
			    #print theta, rho
			    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),1)

		# write the flipped frame
		out.write(frame)
		#cv2.imwrite('asdn.png', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
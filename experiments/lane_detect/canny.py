import cv2
"""
gray = cv2.imread('snap1.png',0)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edges = cv2.Canny(gray,200,300)
cv2.imwrite('edges.png',edges)
"""
cap = cv2.VideoCapture("edited.mp4")

ret, frame = cap.read()
height, width, ch = frame.shape
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('edges.mp4',fourcc, 20.0, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.bilateralFilter(frame, 11, 17, 17)
		edges = cv2.Canny(gray,200,300)
		# write the flipped frame
		img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
		out.write(img)
		#cv2.imwrite('asdn.png', frame)
		#break

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
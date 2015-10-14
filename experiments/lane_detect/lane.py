import cv2
import numpy as np
from math import hypot, degrees, tan, radians

def calculate_distances(pts):
	distances = []
	for i in range(len(pts)-1):
		#row = []
		x0,y0 = pts[i][0],pts[i][1]
		for j in range(i+1,len(pts)):
			x1,y1 = pts[j][0],pts[j][1]
			distances.append([i,j,hypot(x1-x0, y1-y0)])
		#distances.append(row)
	return distances

def find_closest_pt(pts, distances, already_sorted):
	if len(pts) == len(already_sorted):
		return already_sorted
	elif len(already_sorted) == 0:
		already_sorted.append(0)
		return find_closest_pt(pts, distances, already_sorted)
	else:
		from_here = distances[already_sorted[-1]]
		closest = 999999999
		index = -1
		for i in range(len(from_here)):
			if from_here[i][2] < closest and i not in already_sorted:
				index = i
				closest = from_here[i][2]
		already_sorted.append(index)
		return find_closest_pt(pts, distances, already_sorted)

def process_pts(pts, p_array):
	result = []
	for i in p_array:
		x0,y0,w0,t0 = pts[i[0]][0],pts[i[0]][1],pts[i[0]][2],pts[i[0]][4]
		x1,y1,w1,t1 = pts[i[1]][0],pts[i[1]][1],pts[i[1]][2],pts[i[1]][4]
		nx = (x0*w0+x1*w1)/(w0+w1)
		ny = (y0*w0+y1*w1)/(w0+w1)
		nw = w0+w1
		nt = (t0*w0+t1*w1)/(w0+w1)
		result.append([nx,ny,hypot(x0-(-10000),y0),nw,nt])
	for i in range(len(pts)):
		if not pts_exist(i, i, p_array):
			result.append([pts[i][0],pts[i][1],pts[i][2],pts[i][3],pts[i][4]])
	result.sort(key=lambda x:x[2])
	return result

def pts_exist(pt1, pt2, pts):
	for i in pts:
		if pt1 in i or pt2 in i:
			return True
	return False

def get_pts2process(distances, level):
	#print distances
	result = []
	for i in distances:
		#print str(i[0]) + " " + str(i[1]) + " " +  str(any(i[0] or i[1] in row for row in result))
		if i[2] < level and not pts_exist(i[0], i[1], result):
			result.append([i[0], i[1]])
	return result

def abs_angle(radian):
	degree = degrees(radian)
	if degree > 90:
		return radians(degree-180)
	return radians(degree)

def is_close(dist1, dist2, th=0.05):
	s = min(dist1, dist2) #small
	t = max(dist1, dist2) #tall
	try:
		result = (t-s)/float(t)
	except ZeroDivisionError:
		return False
	return result < th

def validify(perms, valid_perms, param=0.005, context=False, cparam=None):

	if is_close(param,0.15, 0.0001):
		print 'param: ' + str(param)
		return valid_perms

	for i in perms:
		x0 = i[0][0]
		x1 = i[1][0]
		x2 = i[2][0]
		x3 = i[3][0]
		if not is_close(x1-x0,x3-x2, param):
			continue
		elif not is_close(x3-x1,x2-x0, param):
			continue
		elif not is_close(x3-x2+x1-x0,2*(x3-x2), param):
			continue
		elif not (x3-x2)*0.63 > x2-x1:
			continue
		elif not (x3-x2)*0.47 < x2-x1:
			continue
		elif not x1 < x2:
			continue
		else:
			valid_perms.append(i)
			perms.remove(i)

	if len(valid_perms) > 5:
		print 'param: ' + str(param)
		return valid_perms
	return validify(perms, valid_perms, param + 0.0025)

def find_lanes(f):
	frame = f.copy()

	height, width, ch = frame.shape
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edges = cv2.Canny(gray,100,200)
	lines = cv2.HoughLines(edges,1,np.pi/180,180)

	pts = []
	if lines is None:
		return frame
	for i in lines:
		for rho,theta in i:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho

			#x1 = int(x0 + 100*(-b))
			#y1 = int(y0 + 100*(a))
			#x2 = int(x0 - 100*(-b))
			#y2 = int(y0 - 100*(a))
			#s_pts = np.array([[x0,y0]], np.int32)
			#frame = cv2.polylines(frame,[s_pts],True,(0,0,255),2)
			pts.append([x0,y0,hypot(x0-(-10000),y0),1,abs_angle(theta)])

	print "points are being analyzed..."

	pts.sort(key=lambda x: x[2])
	starting_point = pts[0]

	distances = calculate_distances(pts)
	distances.sort(key=lambda x:x[2])

	iter_counter = 13

	print "pt count: " + str(len(pts))
	for i in range(iter_counter):
		pts2process = get_pts2process(distances, i)
		pts = process_pts(pts, pts2process)
		distances = calculate_distances(pts)

	for i in pts:
		x0 = i[0]
		y0 = i[1]
		a = np.cos(i[4])
		b = np.sin(i[4])
		y_mov = -y0
		c=x0+y_mov*(-b)
		i[0] = int(c)
		i[1] = int(round(y0+y_mov*(a)))
		i[2] = hypot(i[0], i[1])

	perms = []
	for i in range(len(pts)-3):
		for j in range(i+1, len(pts)-2):
			for k in range(j+1, len(pts)-1):
				for l in range(k+1, len(pts)):
					perms.append([pts[i],pts[j],pts[k],pts[l]])
	
	valid_perms = validify(perms, [])
	print 'valid perms: ' + str(len(valid_perms))
	for j in valid_perms:
		for i in j:
			x0 = i[0]
			y0 = i[1]
			a = np.cos(i[4])
			b = np.sin(i[4])
			x1 = int(round(x0 + 1000*(-b)))
			y1 = int(round(y0 + 1000*(a)))
			x2 = int(round(x0 - 1000*(-b)))
			y2 = int(round(y0 - 1000*(a)))

			y_mov = 10-y0
			x_new = int(round(x0+y_mov*(-b)))
			y_new = int(round(y0+y_mov*(a)))

			s_pts = np.array([[x_new,y_new]], np.int32)
			#blank_image = cv2.polylines(blank_image,[s_pts],True,(0,0,255),3)
			frame = cv2.polylines(frame,[s_pts],True,(0,0,255),3)
			cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
	return frame, valid_perms

def adjust_pts(ps):
	result = []
	for f in range(4):
		prev = 0
		index = 1
		data = []
		indexes = []
		for i in ps:
			values = []
			for j in i:
				x = j[f][0]
				a = j[f][4]
				values.append([x,a])
			pos = 0
			smallest = 999999999
			derivs = [abs(i[0]-prev) for i in values]
			for i in range(len(derivs)):
				if values[i][0] < i:
					pos = i
					smallest = derivs[i]
			if len(derivs) > 0:
				chosen_one = values[pos]
				prev = chosen_one[0]
			else:
				chosen_one = [-1,-1]
			data.append(chosen_one)
			indexes.append(index)
			index+=1

		without_minuses = []
		for i in data:
			if i[0] != -1:
				without_minuses.append(i)

		derivs = [0]
		for i in range(len(without_minuses)-1):
			derivs.append(without_minuses[i+1][0]-without_minuses[i][0])

		for i in range(len(derivs)-1):
			if (derivs[i]>15 and derivs[i+1]<-15) or (derivs[i]<-15 and derivs[i+1]>15):
				without_minuses[i][0] = (without_minuses[i-1][0]+without_minuses[i+1][0])/2
				without_minuses[i][1] = (without_minuses[i-1][1]+without_minuses[i+1][1])/2

		index = 0
		for i in range(len(data)):
			if data[i][0] != -1:
				data[i] = without_minuses[index]
				index+=1

		copy = data[:]

		minus_series = []
		on_track = False
		for i in range(len(data)):
			if data[i][0] == -1:
				if on_track is False:
					on_track = True
					minus_series.append([i,1])
				else:
					minus_series[-1][1]+=1
			else:
				if on_track is True:
					on_track = False

		#print minus_series

		for i in minus_series:
			try:
				last_valid_x = data[i[0]-1][0]
				next_valid_x = data[i[0]+i[1]][0]
				step_x = (next_valid_x-last_valid_x)/(i[1]+1)

				last_valid_a = data[i[0]-1][1]
				next_valid_a = data[i[0]+i[1]][1]
				step_a = (next_valid_a-last_valid_a)/(i[1]+1)
				for j in range(i[1]):
					data[i[0]+j][0] = (step_x*(j+1)) + last_valid_x
					data[i[0]+j][1] = (step_a*(j+1)) + last_valid_a
			except IndexError:
				pass
		result.append(data)
	return result

"""
frame = cv2.imread('lanes/img00037.png')
pts = calculate_pts(frame)
lanes = find_lanes(pts,frame)
cv2.imwrite('houghlines3.jpg', lanes)


t = find_closest_pt(pts, distances,[])


print t
for i in range(len(t)-1):
	x0 = int(pts[t[i]][0]+width)
	y0 = int(pts[t[i]][1]+height)
	x1 = int(pts[t[i+1]][0]+width)
	y1 = int(pts[t[i+1]][1]+height)
	cv2.line(blank_image,(x0,y0),(x1,y1),(0,255,255),1)
cv2.imwrite('lane.png', blank_image)
"""

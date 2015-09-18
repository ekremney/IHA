import cv2
import numpy as np
from math import atan2, cos, sin

SMOOTHING_RADIUS = 30
HORIZONTAL_BORDER_CROP = 20

class TransformParam():
	dx = np.float64(0)
	dy = np.float64(0)
	da = np.float64(0)

	def __init__(self, dx, dy, da):
		self.dx = np.float64(dx)
		self.dy = np.float64(dy)
		self.da = np.float64(da)

class Trajectory():
	x = np.float64(0)
	y = np.float64(0)
	a = np.float64(0)

	def __init__(self, x, y, a):
		x = np.float64(x)
		y = np.float64(y)
		a = np.float64(a)


if __name__ == '__main__':
	cap = cv2.VideoCapture('uav2.mov')

	ret_prev, prev_frame = cap.read()
	prev_frame_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

	max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	last_T = np.array([2,3], np.float64)
	prev_to_cur_transform = []

	while (1):
		ret_cur, cur_frame = cap.read()

		if ret_cur != True:
			break

		cur_frame_grey = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

		prev_corner = cv2.goodFeaturesToTrack(prev_frame_grey, 200, 0.01, 30)
		cur_corner = np.array([], np.float64)

		cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_grey, cur_frame_grey, prev_corner, cur_corner)

		prev_corner2 = []
		cur_corner2 = []

		for i in range(len(status)):
			if status[i] is not False:
				prev_corner2.append(prev_corner[i])
				cur_corner2.append(cur_corner[i])

		prev_corner2 = np.array(prev_corner2, np.float64)
		cur_corner2 = np.array(cur_corner2, np.float64)

		T = cv2.estimateRigidTransform(prev_corner2, cur_corner2, False)

		if T is None:
			T = np.copy(last_T)
		last_T = np.copy(T)

		dx = T[0,2]
		dy = T[1,2]
		da = atan2(T[1,0], T[0,0])

		prev_to_cur_transform.append(TransformParam(dx, dy, da))

		prev_frame = np.copy(cur_frame)
		prev_frame_grey = np.copy(cur_frame_grey)

	a, x, y = 0, 0, 0

	trajectory = []

	for i in prev_to_cur_transform:
		x += i.dx
		y += i.dy
		a += i.da

		trajectory.append(Trajectory(x, y, a))

	smoothed_trajectory = []

	index = 0
	for i in trajectory:
		sum_x, sum_y, sum_a, count = 0, 0, 0, 0

		for j in range(-SMOOTHING_RADIUS, SMOOTHING_RADIUS + 1):
			if index + j >= 0 and index + j < len(trajectory):
				sum_x += trajectory[index+j].x
				sum_y += trajectory[index+j].y
				sum_a += trajectory[index+j].a
				count += 1
		avg_a = sum_a / np.float64(count)
		avg_x = sum_x / np.float64(count)
		avg_y = sum_y / np.float64(count)

		smoothed_trajectory.append(Trajectory(avg_x, avg_y, avg_a))

		index += 1

	new_prev_to_cur_transform = []

	x, y, a = 0, 0, 0

	index = 0
	for i in prev_to_cur_transform:
		x += i.dx
		y += i.dy
		a += i.da

		diff_x = np.float64(smoothed_trajectory[index].x - x)
		diff_y = np.float64(smoothed_trajectory[index].y - y)
		diff_a = np.float64(smoothed_trajectory[index].a - a)

		dx = np.float64(i.dx + diff_x)
		dy = np.float64(i.dy + diff_y)
		da = np.float64(i.da + diff_a)

		new_prev_to_cur_transform.append(TransformParam(dx, dy, da))

	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
	T = np.zeros((2,3), np.float64)
	rows, cols, ch = prev_frame.shape
	vert_border = HORIZONTAL_BORDER_CROP * rows / np.float64(cols)


	output_cap = cv2.VideoWriter('StabilizedVideo.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

	k = 0

	while k < max_frames-1:
		ret_cur, cur_frame = cap.read()

		if ret_cur != True:
			break
		
		T[0,0] = cos(new_prev_to_cur_transform[k].da)
		T[0,1] = -sin(new_prev_to_cur_transform[k].da)
		T[1,0] = sin(new_prev_to_cur_transform[k].da)
		T[1,1] = cos(new_prev_to_cur_transform[k].da)
		T[0,2] = new_prev_to_cur_transform[k].dx
		T[1,2] = new_prev_to_cur_transform[k].dy

		rows, cols, ch = cur_frame.shape
		cur_frame2 = cv2.warpAffine(cur_frame, T, (cols,rows))
		cv2.imwrite("asd.png", cur_frame2)

		cur_frame2 = cur_frame2[vert_border:rows - vert_border, HORIZONTAL_BORDER_CROP:cols - HORIZONTAL_BORDER_CROP]
		cur_frame2 = cv2.resize(cur_frame2, (cols, rows))

		canvas = np.copy(cur_frame2)

		output_cap.write(canvas)

		cv2.waitKey(20)
		k += 1
	output_cap.release()

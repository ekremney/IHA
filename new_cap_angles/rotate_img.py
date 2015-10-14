import cv2, csv
import numpy as np

def read_bboxes(folder, index):
	result = []
	annot_filename = folder + "img{:0>5d}.annot".format(index)
	#annot_filename = folder + "/" + str(index) + ".annot"

	with open(annot_filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		i = 0
		for row in reader:
			result.append([])
			for element in row:
				result[i].append(int(element))
			i += 1
	return result

#[[y1, _, x1, _], [y2, _, x2, _]] = read_bboxes('', 0)

for i in range(0, 350):
	imname = "img{:0>5d}.png".format(i)
	[[y1, _, x1, _], [y2, _, x2, _]] = read_bboxes('', i)
	img = cv2.imread(imname)
		
	rows,cols,ch = img.shape
	pts1 = np.float32([[0,0],[x1,y1],[x2,y2],[0,480]])
	pts2 = np.float32([[0,0],[x1,y1],[x1,y2],[0,480]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(854,480))

	cv2.imwrite(imname, dst)


"""
	imname = "img{:0>5d}.png".format(i)
	[[y1, _, x1, _], [y2, _, x2, _]] = read_bboxes('', i)
	img = cv2.imread(imname)
	
	rows,cols,ch = img.shape
	pts1 = np.float32([[0,0],[x1,y1],[x2,y2],[854,480]])
	pts2 = np.float32([[0,0],[x1,y1],[x1,y2],[854,480]])
	
	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(854,480))

	cv2.imwrite(imname, dst)
"""
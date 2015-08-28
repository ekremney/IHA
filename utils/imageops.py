import cv2, csv
import numpy as np
import time
from random import sample, randint
from libsvm.svmutil import svm_predict
from feature_extractor import extract

def median(mlist):
	mlist.sort()
	length = len(mlist)
	mid = length/2
	if length % 2 == 0:
		return (mlist[mid]+mlist[mid-1])/2
	else:
		return mlist[mid]

def abs_mat(bg_img, img):
	if bg_img.shape != img.shape:
		return False
	else:
		rows, cols = bg_img.shape
		for i in range(rows):
			for j in range(cols):
				if img[i,j] > bg_img[i,j]:
					img[i,j] = img[i,j] - bg_img[i,j]
				else:
					img[i,j] = bg_img[i,j] - img[i,j]
		return img


def img_read(folder, index):
	filename = folder + "/img{:0>5d}.png".format(index)
	return cv2.imread(filename, 0)

def read_bboxes(folder, index):
	result = []
	annot_filename = folder + "/img{:0>5d}.annot".format(index)
	with open(annot_filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		i = 0
		for row in reader:
			result.append([])
			for element in row:
				result[i].append(int(element))
			i += 1
	return result

def read_motion_image(folder, index, bg_img=False):
	motion_threshold = 15
	if bg_img is False:
		# frame difference
		img0 = img_read(folder, index-1)
		img = img_read(folder, index)
		img1 = img_read(folder, index)
		motion_img = abs(img-img0) + abs(img1-img)
	else:
		img = img_read(folder, index)
		motion_img = abs_mat(bg_img,img)

	rows, cols = motion_img.shape
	ret, motion_img = cv2.threshold(motion_img,motion_threshold,255,cv2.THRESH_BINARY)
	motion_img = cv2.medianBlur(motion_img,3)
	return motion_img

def add_bbox_margin(bboxes, marginX, marginY, height, width):
	if bboxes is None:
		return
	for row in bboxes:
		row[0] = max(1, row[0] - marginY)
		row[1] = min(height, row[1] + marginY)
		row[2] = max(1, row[2] - marginX)
		row[3] = min(width, row[3] + marginX)
	return bboxes
'''

bboxes(:,1) = max(1,bboxes(:,1)-marginy);
bboxes(:,2) = min(h,bboxes(:,2)+marginy);
bboxes(:,3) = max(1,bboxes(:,3)-marginx);
bboxes(:,4) = min(w,bboxes(:,4)+marginx);
'''

def img_crop(img, y1, y2, x1, x2):
	img_cut = []
	for i in range(y1, y2):
		img_cut.append([])
		for j in range(x1, x2):
			img_cut[i - y1].append(img[i, j])
	return img_cut


def rand_bbox(bboxes, row, col):
	bb = sample(bboxes, 1)[0]
	width = bb[1] - bb[0] + 1
	height = bb[3] - bb[2] + 1
	maxY = row - height
	maxX = col - width
	y = randint(0, maxY-1)
	x = randint(0, maxX-1)
	neg_bb = [y, y+height-1, x, x+width-1]
	return neg_bb

def overlaps(neg_bb, bboxes, th=0.5):
	does_overlap = 0
	index = 0
	for i in bboxes:
		top = max(i[0], neg_bb[0])
		bottom = min(i[1], neg_bb[1])
		left = max(i[2], neg_bb[2])
		right = min(i[3], neg_bb[3])

		if bottom-top > 0 and right-left > 0:
			intersection = float((bottom-top)*(right-left))

			h1 = neg_bb[1] - neg_bb[0] + 1
			w1 = neg_bb[3] - neg_bb[3] + 1
			h2 = i[1] - i[0] + 1
			w2 = i[3] - i[2] + 1

			union = float(((h1*w1)+(h2*w2)) - intersection)

			if intersection/union > th:
				does_overlap = index
				break
		index += 1

	return does_overlap

def listisize(_list):
	res = []
	for i in _list:
		res.append([i])
	return res


def sliding_window_search(img, motion_img, svm, sbox_height, sbox_width, slide=10, threshold=0.2):
	detections = []
	det_count = 0
	height, width = img.shape

	for i in range(1, (height - sbox_height), slide):
		for j in range(1, (width - sbox_width), slide):
			img_patch = img[i:i+sbox_height-1, j:j+sbox_width-1]
			motion_patch = motion_img[i:i+sbox_height-1, j:j+sbox_width-1]
			img_feat = extract(img_patch, motion_patch)
			y = []
			y.append(0)
			x = []
			x.append(img_feat)
			plabel, acc, pr = svm_predict(y, x, svm, '-b 1')
			if pr[0][0] > threshold:
				detections.append([i, i+sbox_height-1, j, j+sbox_width-1, pr[0][0]])
	return detections


def detect_vehicles(img, motion_img, svm, marginY, marginX):
	jump = 10

	# Sliding window search

	#d1 = sliding_window_search(img, motion_img, svm, 45, 115, jump, 0.8)
	#d1 = sliding_window_search(img, motion_img, svm, 40, 55, jump, 0.85)
	d1 = sliding_window_search(img, motion_img, svm, 30, 45, jump, 0.95)
	
	height, width = img.shape
	d1 = add_bbox_margin(d1, -marginY, -marginX, height, width)
	#d2 = add_bbox_margin(d2, -marginY, -marginX, height, width)
	#d3 = add_bbox_margin(d3, -marginY, -marginX, height, width)
	#d4 = add_bbox_margin(d4, -marginY, marginX, height, width)


	#for i in d2:
	#	d1.append(i)

	#for i in d3:
	#	d1.append(i)

	return d1

#  Felzenszwalb et al.
def non_max_suppression(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1, x2, y1, y2, resemblance, areas = [], [], [], [], [], []
	for i in boxes:
		x1.append(i[2])
		x2.append(i[3])
		y1.append(i[0])
		y2.append(i[1])
		resemblance.append(i[4])
		areas.append((i[3]-i[2])*(i[1]-i[0]))

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	idxs = np.argsort(resemblance)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]

			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / areas[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
	res = []
	for i in pick:
		res.append(boxes[i])
	return res

def compute_detection_AP(detections, bboxes, th=0.4):
	npos = len(bboxes)
	tp = [0] * len(detections)
	for i in range(len(detections)):
		k = overlaps(detections[i], bboxes, th)
		if k != 0:
			tp[i] = 1
			#del bboxes[k]

	pr = [0] * npos
	rc = [0] * npos
	j = 0
	for i in range(len(tp)):
		if tp[i] == 1:
			pr[j] = float(sum(tp[:i]))/(i+1)
			rc[j] = float(j+1)/float(npos)
			j += 1
	ap = np.mean(pr)
	return ap, pr, rc
'''
	for i in range(len(rc)):
		if rc[i] == 0:
			del pr[i] = []
			rc[i] = []
'''
	


'''
function [ap pr rc]=computeDetectionAP(detections,gt, th)
% computes average precision for detections
% gt : ground truth

pr = zeros(npos,1);
rc = zeros(npos,1);
j=0;
for i=1:length(tp)
    if (tp(i)==1)
        j=j+1;
        pr(j)=sum(tp(1:i))/i;
        rc(j)=j/double(npos);
    end
end

ap=mean(pr);

pr(rc==0) = [];
rc(rc==0) = [];




detections = sortrows(detections,-5); % detection probability'e gore descending order'da detectionlari sirala
% non max suppression
i=1;
while (i<=size(detections,1))
    j=i+1;
    while (j<=size(detections,1))
        if ( overlaps(detections(i,:),detections(j,:),th) ~= 0)
            detections(j,:)=[]; % remove jth detection
        else
            j=j+1;
        end        
    end
    i=i+1;
end

end

'''


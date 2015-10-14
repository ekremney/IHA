import cv2, csv

IMAGE_COUNT = 210
FIRST_IMAGE = 100000000

for i in range(IMAGE_COUNT):
	annot_f = str(FIRST_IMAGE + i) + ".annot"
	image_f = str(FIRST_IMAGE + i) + ".png"

	with open(annot_f) as annot_file:
		lines = csv.reader(annot_file)

		img = cv2.imread(image_f)
		for j in lines:
			img = cv2.rectangle(img,(int(j[2]),int(j[0])),(int(j[3]),int(j[1])),(0,255,0),1)
		cv2.imwrite(annot_f + ".png", img)


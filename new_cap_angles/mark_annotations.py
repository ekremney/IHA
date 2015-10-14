import cv2, csv

for i in range(349, 464):
	imname = "img{:0>5d}.png".format(i)
	annotname = "img{:0>5d}.annot".format(i)

	with open(annotname) as annot_file:
		lines = csv.reader(annot_file)

		img = cv2.imread(imname)
		for j in lines:
			img = cv2.rectangle(img,(int(j[2]),int(j[0])),(int(j[3]),int(j[1])),(0,255,0),1)
		cv2.imwrite(annotname+ ".png", img)


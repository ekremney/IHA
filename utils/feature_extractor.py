from skimage.feature import hog as scikit_hog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, exposure
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
import skimage

def a_hog(img):
	bin_n = 16

	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	#print len(ang)
	bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)     # hist is a 64 bit vector
	return hist

def s_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False, normalise=False):
	fd = scikit_hog(image, orientations, pixels_per_cell, cells_per_block, visualise, normalise)
	x = itemfreq(fd.ravel())
	hist = x[:, 1]/sum(x[:, 1])
	return hist

def lbp(image):
	radius = 3 
	no_points = 8 * radius
	lbp_vals = local_binary_pattern(image, no_points, radius, method='uniform')
	x = itemfreq(lbp_vals.ravel())
	hist = x[:, 1]/sum(x[:, 1])
	return hist

def extract(img, motion_img, method='a_hog'):
	result = []
	resized_img = cv2.resize(img, (5, 5)) 
	resized_motion_img = cv2.resize(motion_img, (5, 5))
	img_norm = np.linalg.norm(img)
	motion_img_norm = np.linalg.norm(motion_img)
	rows, cols = img.shape
	area = rows*cols

	#motion_img = motion_img / float(motion_img_norm);
	#img = img / float(img_norm);

	if method == 's_hog':
		img_hog = s_hog(img)
		motion_img_hog = s_hog(motion_img)
	if method == 'a_hog':
		img_hog = a_hog(img)
		motion_img_hog = a_hog(motion_img)
	if method == 'lbp':
		img_hog = lbp(img)
		motion_img_hog = lbp(img)
	if method == 'hybrid':
		img_hog = lbp(img)
		motion_img_hog = lbp(motion_img)

	for i in img_hog:
		result.append(i)

	for i in motion_img_hog:
		result.append(i)

	for i in range(5):
		for j in range(5):
			result.append(resized_img[i, j])

	for i in range(5):
		for j in range(5):
			result.append(resized_motion_img[i, j])
	result.append(img_norm)
	result.append(motion_img_norm)
	result.append(rows)
	result.append(cols)
	result.append(area)
	return result


#print skimage.__file__
#img = cv2.imread('../mi1.png', 0)
#fd = scikit_hog(img) #lbp(img)
#print len(fd[0])

#fd = extract(img, img)
#hog = cv2.HOGDescriptor()
#print hog.compute.__doc__
#fd = hog.compute(img)

#print "fd:"
#print len(fd)


'''
for i in fd:
	print i

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
'''

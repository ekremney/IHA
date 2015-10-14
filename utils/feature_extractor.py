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
	bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)     # hist is a 64 bit vector
	return hist

def s_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False, normalise=False):
	fd = scikit_hog(image, orientations, pixels_per_cell, cells_per_block, visualise, normalise)
	fd = fd.ravel()
	hist, bin_edges = np.histogram(fd, bins = np.linspace(0, 1, 100))
	return hist

def lbp(image):
	radius = 3 
	no_points = 8 * radius

	"""
	method : {'default', 'ror', 'uniform', 'var'}
        Method to determine the pattern.
        * 'default': original local binary pattern which is gray scale but not
            rotation invariant.
        * 'ror': extension of default implementation which is gray scale and
            rotation invariant.
        * 'uniform': improved rotation invariance with uniform patterns and
            finer quantization of the angular space which is gray scale and
            rotation invariant.
        * 'nri_uniform': non rotation-invariant uniform patterns variant
            which is only gray scale invariant [2]_.
        * 'var': rotation invariant variance measures of the contrast of local
            image texture which is rotation but not gray scale invariant.
	"""

	lbp_vals = local_binary_pattern(image, no_points, radius, method='uniform')
	d = lbp_vals.ravel()
	hist, bin_edges = np.histogram(d, bins = np.linspace(0, 25, 26))
	return hist

def test(img):
	hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	result = hog.detectMultiScale(img, **hogParams)
	return result

def extract(img, motion_img, method, feature='all'):
	result = []
	
	if feature == 'img' or feature == 'all':
		resized_img = cv2.resize(img, (5, 5)) 
		img_norm = np.linalg.norm(img)

	if feature == 'motion' or feature == 'all':
		resized_motion_img = cv2.resize(motion_img, (5, 5))
		motion_img_norm = np.linalg.norm(motion_img)

	rows, cols = img.shape
	area = rows*cols

	#motion_img = motion_img / float(motion_img_norm);
	#img = img / float(img_norm);

	
	if feature == 'img' or feature == 'all':
		img_hog = globals()[method](img)
		[result.append(i) for i in img_hog]
		[result.append(i) for row in resized_img for i in row]
		result.append(img_norm)
	if feature == 'motion' or feature == 'all':
		motion_img_hog = globals()[method](motion_img)
		[result.append(i) for i in motion_img_hog]
		[result.append(i) for row in resized_motion_img for i in row]
		result.append(motion_img_norm)

	result.append(rows)
	result.append(cols)
	result.append(area)
	
	return result


#print skimage.__file__
"""
img = cv2.imread('asd.png', 0)
d = s_hog(img)
print d
"""
"""
print max(d)
print min(d)
hist, bin_edges = np.histogram(d, bins = np.linspace(-0.0002, 0.001, 100))
print hist



d = d.ravel()
print len(d)
hist, bin_edges = np.histogram(d, bins = range(27))
print hist
print bin_edges
for i,j in zip(bin_edges, hist):
	print str(i) + "- " + str(j)
"""
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
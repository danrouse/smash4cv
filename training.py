import numpy as np
import cv2

from SmashCVCore import processDigit

# Generate training data
np.random.seed(2099)
responses = [int(i * 10) for i in np.random.random((2000, 1))]
samples = np.empty((0, 10 * 10))

np.savetxt('templates/training-responses.txt', responses, '%d')

im = cv2.imread('templates/training.png', 0)
cv2.imshow('base', im)
cv2.waitKey(0)

im_h = im.shape[0]
im_w = im.shape[1]
char_w = (im_w / 2000)

for i in range(0, 2000):
	x1 = max(0, (i*char_w) + 2)
	x2 = min(im_w, ((i+1)*char_w) + 2)

	# crop out single char and process
	char = im[0:im_h, x1:x2]
	char_resized = processDigit(char)

	# insert 1-dimensional sample
	sample = char_resized.reshape((1, 10 * 10))
	samples = np.append(samples, sample, 0)

	cv2.imshow('OCR-A', char)
	cv2.imshow('OCR-B', char_resized)
	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
np.savetxt('templates/training-samples.data', samples, '%d')
np.savetxt('templates/training-responses.data', responses, '%d')
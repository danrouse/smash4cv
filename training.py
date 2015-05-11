import numpy as np
import cv2
import os

if(os.path.exists('templates/training-responses.data')):
	os.remove('templates/training-responses.txt')
	os.remove('templates/training-responses.data')
	os.remove('templates/training-samples.data')

from SmashCVCore import processDigit

DEBUG = False

# Generate training data
np.random.seed(4444)
responses = [int(i * 10) for i in np.random.random((2000, 1))]
samples = np.empty((0, 10 * 10))

np.savetxt('templates/training-responses.txt', responses, '%d')

im = cv2.imread('templates/training.png', 0)
im_h = im.shape[0]
im_w = im.shape[1]
char_w = (im_w / 2000)

for i in range(0, 2000):
	x1 = max(0, (i*char_w))
	x2 = min(im_w, ((i+1)*char_w))

	# add a little noise
	#off_x = int(round(np.random.rand() * 4)) - 3
	off_x = 0
	off_top = int(round(np.random.rand()))
	#off_bot = int(round(np.random.rand())) * -1
	off_bot = 0

	if(responses[i] == 1):
		off_x += int(np.random.rand() * 4)

	# crop out single char and process
	char = im[off_top:im_h+off_bot, x1+off_x:x2 - 2]
	char_resized = cv2.resize(char, (10, 10))

	# insert 1-dimensional sample
	sample = char_resized.reshape((1, 10 * 10))
	samples = np.append(samples, sample, 0)

	if(DEBUG):
		cv2.imshow('OCR-A', char)
		cv2.imshow('OCR-B', char_resized)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		elif key & 0xFF == ord('w'):
			cv2.waitKey(0)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
np.savetxt('templates/training-samples.data', samples, '%d')
np.savetxt('templates/training-responses.data', responses, '%d')
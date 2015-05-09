import cv2
import numpy as np

# Templates
tpl_zero = cv2.imread('templates/zero-percent-color-small.png', 0)
tpl_percent = cv2.imread('templates/percent-sign.png', 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# OCR KNearest
samples = np.loadtxt('templates/training-samples.data', np.float32)
responses = np.loadtxt('templates/training-responses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.KNearest()
model.train(samples, responses)

# Image processing options
calibThreshold = 0.7

# State
STATE_CALIBRATE = 1
STATE_INGAME = 2
STATE_CHARSELECT = 3
STATE_MAPSELECT = 4

def log(message):
	print message

def isolateChannel(im, channel):
	im_channel = cv2.split(im)[channel]
	im_buf = np.zeros_like(im)
	
	for i in range(0,3):
		im_buf[:,:,i] = im_channel

	return cv2.cvtColor(im_buf, cv2.COLOR_BGR2GRAY)

def calibrateFrame(src_im):
	# TODO: Look for character selection, map selection

	# Look for "0%"
	matches = cv2.matchTemplate(src_im, tpl_zero, cv2.TM_CCOEFF_NORMED)
	matches = np.where(matches > calibThreshold)
	matches = zip(*matches[::-1])

	if(len(matches) > 0):
		# Round each X down to nearest 2
		# cast to a set (removing dupes), back to a list (ordered), and sorted
		xs = sorted(list(set([int(round(pt[0] / 2, 0) * 2) \
				for pt in matches])))
		y = matches[0][1]
		ROIs = [(x, y) for x in xs]
		return STATE_INGAME, ROIs

	return STATE_CALIBRATE, []

def processDigit(src_im):
	im_eq = cv2.equalizeHist(src_im)
	_res, im_bin = cv2.threshold(im_eq, 100, 255, cv2.THRESH_BINARY)
	#im_morph = cv2.erode(im_bin, kernel, 1)
	im_border = cv2.copyMakeBorder(im_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])
	im_resize = cv2.resize(im_border, (10, 10))
	return im_resize

def damageOCR(src_im, ROIs):
	# Text is centered at (21, ),
	# Characters have a width of 10
	digits = []

	for roi_i, loc in enumerate(ROIs):
		roi_digits = []

		# Crop ROI from source
		roi_im = src_im[
	 		loc[1] + 3: loc[1] + tpl_zero.shape[1] - 2,		# TODO: Magic numbers
	 		loc[0] - 13 : loc[0] + tpl_zero.shape[0] + 15]
	 	roi_dim = roi_im.shape[::-1]

		# Find percent sign to figure digit count
		pct_im = roi_im[
			(roi_dim[1] / 2):roi_dim[1],
			(roi_dim[0] / 2)+1:roi_dim[0]]

		_res, pct_im_bin = cv2.threshold(pct_im, 80, 255, cv2.THRESH_BINARY)
		matches = cv2.matchTemplate(pct_im_bin, tpl_percent, cv2.TM_CCOEFF_NORMED)
		_res, match_val, _res, match_loc = cv2.minMaxLoc(matches)
		if(match_val < 0.3):
			# print '%d | NO Confidence: %0.2f' % (match_loc[0], match_val)
			continue

		pct_x = match_loc[0]

		num_digits = 1
		if(pct_x > 10):		# TODO: Magic numbers
			num_digits = 3
		elif(pct_x > 5):
			num_digits = 2

		# adjust relative to ROI
		pct_x += (roi_dim[0] / 2) + 1

		for i in range(num_digits):
			# crop digit from ROI based on offset from percentage
			digit_im = roi_im[
				0:roi_dim[1],
				pct_x - (10 * (i + 1)) + 2 : pct_x - (10 * i)	# TODO: Magic numbers
				]

			# OCR
			digit_im = processDigit(digit_im)
			digit_1d = digit_im.reshape((1, 10 * 10))
			digit_1d = np.float32(digit_1d)
			digit, _res, _res, conf = model.find_nearest(digit_1d, k=1)
			roi_digits.insert(0, (digit, conf))
			#cv2.imshow('ROI %d, Digit %i' % (roi_i, i), digit_im)

		#digits.append(roi_digits)
		digits.append([d[0] for d in roi_digits])

	print digits

def processVideo(video_path):
	video = cv2.VideoCapture(video_path)
	state = STATE_CALIBRATE
	regions = []

	while(video.isOpened()):
		ret, frame = video.read()
		if not ret: break

		gray = isolateChannel(frame, 2)

		if state == STATE_CALIBRATE:
			state, regions = calibrateFrame(gray)

		# If calibration was unsuccessful, pass
		if state == STATE_CALIBRATE:
			log('No calibration found')
			continue
		elif state == STATE_INGAME:
			percents = damageOCR(gray, regions)

		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		elif key & 0xFF == ord('w'):
			cv2.waitKey(0)

		cv2.imshow('frame', frame)
		cv2.imshow('gray', gray)

	video.release()

	cv2.destroyAllWindows()
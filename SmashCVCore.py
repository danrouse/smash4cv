import cv2
import numpy as np

# Templates
tpl_char = cv2.imread('templates/char-select.png', 0)
tpl_stage = cv2.imread('templates/stage-select.png', 0)
tpl_zero = cv2.imread('templates/zero-percent-color-small.png', 0)
tpl_percent = cv2.imread('templates/percent-sign.png', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# OCR KNearest
try:
	samples = np.loadtxt('templates/training-samples.data', np.float32)
	responses = np.loadtxt('templates/training-responses.data', np.float32)
	responses = responses.reshape((responses.size, 1))
	model = cv2.KNearest()
	model.train(samples, responses)
except IOError:
	pass

# Image processing options
match_threshold = 0.75
pct_threshold = 0.6

# State
STATE_CALIBRATE = 1
STATE_INGAME = 2
STATE_CHARSELECT = 3
STATE_STAGESELECT = 4

def log(message):
	print message

def isolateChannel(im, channel):
	im_channel = cv2.split(im)[channel]
	im_buf = np.zeros_like(im)
	
	for i in range(0,3):
		im_buf[:,:,i] = im_channel

	return cv2.cvtColor(im_buf, cv2.COLOR_BGR2GRAY)

def reduceRound(list_in, round_by):
	# Round each X down
	# cast to a set (removing dupes), back to a list (ordered), and sorted
	return sorted(list(set([int(round(pt[0] / round_by, 0) * round_by) \
					for pt in list_in])))

def filterMatchTemplate(src_im, tpl_im, threshold, reduce_to_x=False, round_by=2):
	matches = cv2.matchTemplate(src_im, tpl_im, cv2.TM_CCOEFF_NORMED)
	matches = np.where(matches > threshold)
	matches = zip(*matches[::-1])

	if(reduce_to_x and len(matches) > 0):
		xs = reduceRound(matches, round_by)
		y = matches[0][1]
		return [(x, y) for x in xs]
	else:
		return matches

def calibrateFrame(src_im):
	im_color = cv2.cvtColor(src_im, cv2.COLOR_GRAY2BGR)

	# Search for character selection: gamepad icon for each player
	cs_matches = filterMatchTemplate(src_im, tpl_char, match_threshold, True)
	if(len(cs_matches) > 0):
		return STATE_CHARSELECT, cs_matches

	# Look for random button on stage selection
	ss_matches = filterMatchTemplate(src_im, tpl_stage, match_threshold)
	if(len(ss_matches) > 0):
		return STATE_STAGESELECT, ss_matches

	# Look for "0%" for in-game initialization
	g_matches = filterMatchTemplate(src_im, tpl_zero, match_threshold, True, 3)
	if(len(g_matches) > 1):
		return STATE_INGAME, g_matches

	return STATE_CALIBRATE, []

def processDigit(src_im, equalize=True, thresh=100):
	#if(equalize):
	#	src_im = cv2.equalizeHist(src_im)
	#_res, im_bin = cv2.threshold(src_im, thresh, 255, cv2.THRESH_BINARY)
	#im_border = cv2.copyMakeBorder(im_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])
	# if(erode):
	# 	im_bin = cv2.erode(im_bin, kernel, 1)
	im_resize = cv2.resize(src_im, (10, 10))
	return im_resize

def damageOCR(src_im, ROIs):
	gray_im = isolateChannel(src_im, 2)

	# Text is centered at (21, ),
	# Characters have a width of 10
	digits = []

	for roi_i, loc in enumerate(ROIs):
		roi_digits = []

		# Crop ROI from source
		roi_im = gray_im[
	 		loc[1] + 1: loc[1] + tpl_zero.shape[1] - 3,		# TODO: Magic numbers
	 		loc[0] - 13 : loc[0] + tpl_zero.shape[0] + 15]
	 	roi_dim = roi_im.shape[::-1]
	 	#cv2.imshow('ROI-%d' % roi_i, roi_im)

		# Find percent sign to figure digit count
		pct_im = roi_im[
			(roi_dim[1] / 2):roi_dim[1],
			(roi_dim[0] / 2)+1:roi_dim[0]]

		#_res, pct_im_bin = cv2.threshold(pct_im, 80, 255, cv2.THRESH_BINARY)
		matches = cv2.matchTemplate(pct_im, tpl_percent, cv2.TM_CCOEFF_NORMED)
		_res, match_val, _res, match_loc = cv2.minMaxLoc(matches)

		if(match_val > pct_threshold):
			pct_x = match_loc[0]

			num_digits = 1
			if(pct_x > 9):		# TODO: Magic numbers
				num_digits = 3
			elif(pct_x > 4):
				num_digits = 2

			# adjust relative to ROI
			pct_x += (roi_dim[0] / 2) + 1

			for i in range(num_digits):
				# crop digit from ROI based on offset from percentage
				digit_im = roi_im[
					1:roi_dim[1],
					pct_x - (10.5 * (i + 1)) + 2 : pct_x - (10 * i) ]	# TODO: Magic numbers

				# kNN OCR
				ocr_thresh = 210 - (40 * num_digits)
				digit_im = processDigit(digit_im)
				digit_1d = digit_im.reshape((1, 10 * 10))
				digit_1d = np.float32(digit_1d)
				digit, _res, _res, conf = model.find_nearest(digit_1d, k=1)				

				roi_digits.insert(0, (int(digit), int(conf)))

			digits_str = ''.join([str(d[0]) for d in roi_digits])
			digits_conf = ','.join([str(d[1]) for d in roi_digits])

			cv2.putText(src_im, digits_str, loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.putText(src_im, '%0.2f' % match_val, (loc[0], loc[1] + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.putText(src_im, digits_conf, (loc[0], loc[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			digits.append(digits_str)
		else:
			digits.append('')
			#digits.append(str(reduce(lambda x: int(x[0]), roi_digits)))

	return digits

def processVideo(video_path):
	video = cv2.VideoCapture(video_path)
	#video.set(0, 3000)
	regions = []

	state = STATE_CALIBRATE
	prev_state = STATE_CALIBRATE
	events = []
	
	frames_without_digits = 0
	total_frames = video.get(7)
	prev_percent = 0

	while(video.isOpened()):
		ret, frame = video.read()
		if not ret: break

		frame_ms = video.get(0)
		frame_progress = int((video.get(1) / total_frames) * 100)
		if(frame_progress != prev_percent):
			print '%d%%...' % prev_percent

		if state != STATE_INGAME:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			state, regions = calibrateFrame(gray)

		# If calibration was unsuccessful, pass
		if state == STATE_CALIBRATE:
			#log('No calibration found')
			continue

		# Track state changes
		if prev_state != state:
			events.append((frame_ms, state, []))
			print 'State change: %d ms, %d' % (frame_ms, state)

		prev_state = state
		prev_percent = frame_progress

		#if state == STATE_CHARSELECT:
		#elif state == STATE_STAGESELECT:
		#elif state == STATE_INGAME:
		if state == STATE_INGAME:
			digits = damageOCR(frame, regions)
			if(''.join(digits) == ''):
				frames_without_digits += 1
				if(frames_without_digits > 100):
					state = STATE_CALIBRATE
			elif len(events) < 1 or digits != events[-1][2]:
				events.append((frame_ms, state, digits))

		# key = cv2.waitKey(1)
		# if key & 0xFF == ord('q'):
		# 	break
		# elif key & 0xFF == ord('w'):
		# 	cv2.waitKey(0)

		#cv2.imshow('frame', frame)
		#cv2.imshow('gray', gray)

	video.release()
	print events
	#cv2.destroyAllWindows()
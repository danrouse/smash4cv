import cv2
import numpy as np
import tesseract

api = tesseract.TessBaseAPI()
api.Init('.', 'eng', tesseract.OEM_DEFAULT)
api.SetPageSegMode(tesseract.PSM_AUTO)
#api.SetVariable('tessedit_char_whitelist', '0123456789')

DEBUG = False

# Templates
tpl_char = cv2.imread('templates/char-select.png', 0)
tpl_stage = cv2.imread('templates/stage-select.png', 0)
tpl_zero = cv2.imread('templates/zero-percent-color-small.png', 0)
tpl_percent = cv2.imread('templates/percent-sign.png', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))

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
# Lots of magic numbers here
# Does making them constants remove the magic?
# I just want to believe
MATCH_THRESHOLD = 0.75
MATCH_THRESHOLD_ZERO = 0.65
MATCH_THRESHOLD_PCT = 0.6
PCT_X_TWO_DIGITS = 5
PCT_X_THREE_DIGITS = 10
CROP_DAMAGE = (1, -2, -13, 15)
CROP_DIGIT = (2, 0)
DIGIT_WIDTH = 10
FRAME_TIMEOUT = 300
KNN_CONF_THRESHOLD = 350000
OCR_CONF_THRESHOLD = 0.7
AREA_STAGE_TEXT = (1000, )

# State
STATE_CALIBRATE = 1
STATE_INGAME = 2
STATE_CHARSELECT = 3
STATE_STAGESELECT = 4

def reduce_round(list_in, round_by):
	"""Convert a list of (x,y) tuples to a sorted list of rounded, unique values.

	Args:
		list_in (list): List of (x,y) tuples
		round_by (int): Amount to round each X value by
	"""

	# Round each X down
	# cast to a set (removing dupes), back to a list (ordered), and sorted
	tick = dt()
	ret = sorted(list(set([int(round(pt[0] / round_by, 0) * round_by) \
					for pt in list_in])))
	dt('reduce_round', tick)
	return ret

def filtered_match(src_im, tpl_im, threshold, reduce_to_x=False, round_by=2):
	"""Match a template and only return results above a specified threshold.

	Args:
		src_im (cv2 image): Frame to process
		tpl_im (cv2 image): Template to find
		threshold (float): Confidence threshold
		reduce_to_x (bool, optional): Reduce matches to rounded x values by a single y
		round_by (int, optional): Reduced X values are rounded by this amount

	Returns:
		matches (list): List of matched (x, y) coordinates
	"""

	tick = dt()
	matches = cv2.matchTemplate(src_im, tpl_im, cv2.TM_CCOEFF_NORMED)
	dt('filter mt', tick)

	t_a = dt()
	matches = np.where(matches > threshold)
	matches = zip(*matches[::-1])
	dt('filter s2', t_a)

	if(reduce_to_x and len(matches) > 0):
		xs = reduce_round(matches, round_by)
		y = matches[0][1]
		return [(x, y) for x in xs]
	else:
		return matches

def dt(title='',c=0):
	"""Measure performance.

	Example:
		>>> timer = dt()
		>>> [do_something(i) for i in range(100)]
		>>> dt('foo', timer)
		'foo: 0.025ms'

	Args:
		title (str, optional): Timer name for output
		c (int, optional): Initial timer value

	Returns:
		tick (int): Initial timer value, returned when called without arguments
	"""

	if not DEBUG: return

	tick = cv2.getTickCount()
	if c>0:
		diff = (tick - c) / cv2.getTickFrequency()
		print '%s: %.3fms' % (title, diff * 1000)
	else:
		return tick

def calibrate_frame(src_im):
	"""Detect which state a frame is in.

	Uses cv2.matchTemplate to look for, in the following order:
		In the top-left quadrant of the frame:
			- back button from character select screen
			- back button from stage select screen
		In the entire frame:
			- '0%' for each player

	Args:
		src_im (cv2 image): The frame to process

	Returns:
		state (int): Constant of detected state, one of
			STATE_CHARSELECT, STATE_STAGESELECT, STATE_INGAME,
			or, if nothing was found, STATE_CALIBRATE
		results (list): Filtered ROI (x, y) positions
	"""
	tick = dt()

	state = STATE_CALIBRATE
	results = []
	im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)

	# Crop the top-left quadrant for faster searching
	im_cropped = im_gray[0:im_gray.shape[0]/2, 0:im_gray.shape[1]/2]

	# Search for character selection: gamepad icon for each player
	cs_matches = filtered_match(im_cropped, tpl_char, MATCH_THRESHOLD, True)
	if(len(cs_matches) > 0):
		state = STATE_CHARSELECT
		results = cs_matches
	else:
		# Look for random button on stage selection
		ss_matches = filtered_match(im_cropped, tpl_stage, MATCH_THRESHOLD)
		if(len(ss_matches) > 0):
			state = STATE_STAGESELECT
			results = ss_matches
		else:
			# Look for "0%" for in-game initialization
			g_matches = filtered_match(im_gray, tpl_zero, MATCH_THRESHOLD_ZERO, True, 3)
			if(len(g_matches) > 1):
				state = STATE_INGAME
				results = g_matches

	dt('calibrate', tick)
	return state, results

def read_stage(src_im, conf_threshold=OCR_CONF_THRESHOLD):
	"""Read stage name from a frame

	Args:
		src_im (cv2 image): The frame to process
		conf_threshold (int, optional): Tesseract OCR confidence threshold

	Returns:
		stage (string): stage name if OCR is successful, '' otherwise
	"""

	# Crop out bottom-left quadrant
	im_cropped = src_im[
		src_im.shape[0]/2:src_im.shape[0],
		0:src_im.shape[1]/2]
	im_gray = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2GRAY)

	# Severe thresholding to isolate white text
	_res, im_thresh = cv2.threshold(im_gray, 240, 255, cv2.THRESH_BINARY)
	im_morph = cv2.erode(im_thresh, kernel, 3)
	im_morph_2 = cv2.dilate(im_morph, kernel_h, 5)
	im_morph_3 = cv2.dilate(im_morph, kernel_h, 5)
	contours, _res = cv2.findContours(im_morph_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	largest_area = 0
	largest_bbox = []
	largest_contour = []
	for c in contours:
		area = cv2.contourArea(c)
		if area > largest_area:
			largest_area = area
			largest_bbox = cv2.boundingRect(c)
			largest_contour = c

	if largest_area == 0:
		# no matches in sight
		return ''

	cv2.drawContours(im_cropped, largest_contour, -1, (0, 255, 0), -1)
	print 'area: %0.2f' % largest_area
	print largest_bbox
	text_im = im_gray[
		largest_bbox[1]:largest_bbox[1] + largest_bbox[3],
		largest_bbox[0]:largest_bbox[0] + largest_bbox[2]]
	_res, text_im = cv2.threshold(text_im, 200, 255, cv2.THRESH_BINARY)
	text_im = cv2.dilate(text_im, kernel, 2)

	# Tesseract OCR
	ocr_im = cv2.cv.CreateImageHeader(text_im.shape[::-1], cv2.cv.IPL_DEPTH_8U, 1)
	cv2.cv.SetData(ocr_im, text_im.tostring(), text_im.dtype.itemsize * text_im.shape[1])
	tesseract.SetCvImage(ocr_im, api)
	ocr_text = api.GetUTF8Text()
	ocr_conf = api.MeanTextConf()

	print 'Stage: %s (%d%%)' % (ocr_text.strip(), ocr_conf)
	cv2.imshow('ocr', text_im)
	cv2.imshow('ocr-search', im_morph_3)
	cv2.imshow('ocr-full', im_cropped)
	cv2.waitKey(0)

	if ocr_conf > conf_threshold:
		return ocr_text.strip()
	else:
		return ''

def read_digits(src_im, ROIs, prev_result, conf_threshold=KNN_CONF_THRESHOLD):
	"""Read digits from specified ROIs in a frame.

	Args:
		src_im (cv2 image): The frame to process
		ROIs (list): List of (x, y) coordinates to watch
		prev_result (list, optional): list of results from previous frame,
			for low-confidence value replacement
		conf_threshold (int, optional): k-NN confidence upper bound

	Returns:
		digits (list): OCRed integers corresponding to each input ROI
		amount_replaced (float): Proportion of low-confidence values replaced
	"""

	digits = []
	amount_replaced = 0.0

	# Isolate red channel
	im_channel = cv2.split(src_im)[2]
	im_buf = np.zeros_like(src_im)
	for i in range(0,3):
		im_buf[:,:,i] = im_channel
	gray_im = cv2.cvtColor(im_buf, cv2.COLOR_BGR2GRAY)

	for roi_i, loc in enumerate(ROIs):
		roi_digits = []

		# Crop ROI from source
		roi_im = gray_im[
	 		loc[1] + CROP_DAMAGE[0]: loc[1] + tpl_zero.shape[1] + CROP_DAMAGE[1],
	 		loc[0] + CROP_DAMAGE[2] : loc[0] + tpl_zero.shape[0] + CROP_DAMAGE[3]]
	 	roi_dim = roi_im.shape[::-1]

		# Find percent sign to figure digit count
		pct_im = roi_im[
			(roi_dim[1] / 2):roi_dim[1],
			(roi_dim[0] / 2)+1:roi_dim[0]] # Crop left x by 1

		matches = cv2.matchTemplate(pct_im, tpl_percent, cv2.TM_CCOEFF_NORMED)
		_res, match_val, _res, match_loc = cv2.minMaxLoc(matches)

		if(match_val > MATCH_THRESHOLD_PCT):
			pct_x = match_loc[0]

			num_digits = 1
			if(pct_x >= PCT_X_THREE_DIGITS):
				num_digits = 3
			elif(pct_x >= PCT_X_TWO_DIGITS):
				num_digits = 2

			# adjust relative to ROI
			pct_x += (roi_dim[0] / 2) + 1

			for i in range(num_digits):
				# crop digit from ROI based on offset from percentage
				digit_im = roi_im[
					1:roi_dim[1],
					pct_x - (DIGIT_WIDTH * (i + 1)) + CROP_DIGIT[0] : pct_x - (DIGIT_WIDTH * i) + CROP_DIGIT[1] ]

				# Tesseract OCR
				# ocr_im = cv2.cv.CreateImageHeader(digit_im.shape[::-1], cv2.cv.IPL_DEPTH_8U, 1)
				# cv2.cv.SetData(ocr_im, digit_im.tostring(), digit_im.dtype.itemsize * digit_im.shape[1])
				# tesseract.SetCvImage(ocr_im, api)
				# ocr_text = api.GetUTF8Text()
				# ocr_conf = api.MeanTextConf()

				# kNN OCR
				digit_im = cv2.resize(digit_im, (10, 10))
				digit_1d = digit_im.reshape((1, 10 * 10))
				digit_1d = np.float32(digit_1d)
				digit, _res, _res, conf = model.find_nearest(digit_1d, k=1)

				# print 'KNN: %d (%d) | Tess: %s (%d%%)' % (int(digit), int(conf), ocr_text, ocr_conf)

				if conf < conf_threshold: # confidence is inverse with kNN
					roi_digits.append(int(digit))
				else:
					# use result from previous, if exists
					prev_roi = str(prev_result[roi_i])
					prev_roi_len = len(prev_roi)
					if prev_roi_len > i:
						#print 'append prev: repl %d with %s' % (int(digit), prev_roi[prev_roi_len - i - 1])
						roi_digits.append(prev_roi[prev_roi_len - i - 1])

					amount_replaced += (0.75 / num_digits) / len(ROIs)

			#digits_str = ''.join([str(d) for d in roi_digits])
			# digits_conf = ','.join([str(d[1]) for d in roi_digits])
			digits.append(int(''.join(str(d) for d in roi_digits[::-1])))

			# cv2.putText(src_im, digits_str, loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# cv2.putText(src_im, '%0.2f' % match_val, (loc[0], loc[1] + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# cv2.putText(src_im, digits_conf, (loc[0], loc[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			
		else:
			# use result from previous, if exists
			if len(prev_result) > roi_i:
				digits.append(prev_result[roi_i])
				amount_replaced += 1.0 / len(ROIs)
			else:
				digits.append([])

	return digits, amount_replaced

def process_video(video_path):
	"""Process an entire video.

	Args:
		video_path (str): Path to local video to load

	Returns:
		events (list): List of detected events and event data
	"""

	video = cv2.VideoCapture(video_path)
	regions = []

	state = STATE_CALIBRATE
	prev_state = STATE_CALIBRATE
	events = {2: [], 3: [], 4: []}
	
	frames_without_digits = 0
	total_frames = video.get(7)
	prev_percent = 0

	while(video.isOpened()):
		tick = dt()

		ret, frame = video.read()
		if not ret: break

		frame_ms = int(video.get(0))
		frame_num = video.get(1)

		# See progress in CLI
		frame_progress = int((frame_num / total_frames) * 100)
		if(frame_progress != prev_percent):
			print '%d%%...' % prev_percent
		prev_percent = frame_progress

		# Calibrate if not tracking for OCR
		if state != STATE_INGAME:
			state, regions = calibrate_frame(frame)

		# If calibration was unsuccessful, skip 15 frames
		if state == STATE_CALIBRATE:
			video.set(1, frame_num + 15)
			continue

		# Track state changes
		if prev_state != state:
			#events.append((frame_ms, state, []))
			events[state].append((frame_ms, []))
			print 'State change: %d ms, %d' % (frame_ms, state)
		prev_state = state

		#if state == STATE_CHARSELECT:
		if state == STATE_STAGESELECT:
			# Crop bottom left quadrant to OCR stage name
			stage = read_stage(frame)
			#print 'Stage: %s' % stage
		elif state == STATE_INGAME:
			digits, replaced = read_digits(frame, regions, events[STATE_INGAME][-1][1])
			OCR_CONF_THRESHOLD = 0.7
			
			if(replaced == 1.0):
				frames_without_digits += 1
				if(frames_without_digits > FRAME_TIMEOUT):
					state = STATE_CALIBRATE
					frames_without_digits = 0
					print 'Game state lost: %d ms' % frame_ms
			else:
				# append only if digits changed
				if digits != events[STATE_INGAME][-1][1]:
					events[STATE_INGAME].append((frame_ms, digits))
				# else:
				# 	# append first event unconditionally
				# 	events[STATE_INGAME].append((frame_ms, digits))

		# cv2.imshow('frame', frame)
		# key = cv2.waitKey(1)
		# if key & 0xFF == ord('q'):
		# 	break
		# elif key & 0xFF == ord('w'):
		# 	cv2.waitKey(0)

		dt('frame', tick)

	video.release()
	return events
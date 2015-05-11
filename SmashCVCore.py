import cv2
import numpy as np

DEBUG = False

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
	tick = dt()
	ret = sorted(list(set([int(round(pt[0] / round_by, 0) * round_by) \
					for pt in list_in])))
	dt('reduceRound', tick)
	return ret

def filterMatchTemplate(src_im, tpl_im, threshold, reduce_to_x=False, round_by=2):
	tick = dt()
	matches = cv2.matchTemplate(src_im, tpl_im, cv2.TM_CCOEFF_NORMED)
	dt('filter mt', tick)

	t_a = dt()
	matches = np.where(matches > threshold)
	matches = zip(*matches[::-1])
	dt('filter s2', t_a)

	if(reduce_to_x and len(matches) > 0):
		xs = reduceRound(matches, round_by)
		y = matches[0][1]
		return [(x, y) for x in xs]
	else:
		return matches

def dt(title='',c=0):
	if not DEBUG: return

	tick = cv2.getTickCount()
	if c>0:
		diff = (tick - c) / cv2.getTickFrequency()
		print '%s: %.3fms' % (title, diff * 1000)
	else:
		return tick

def calibrateFrame(src_im):
	tick = dt()

	state = STATE_CALIBRATE
	results = []
	im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)

	# Search for character selection: gamepad icon for each player
	t_a = dt()
	cs_matches = filterMatchTemplate(im_gray, tpl_char, match_threshold, True)
	dt('mt char', t_a)
	if(len(cs_matches) > 0):
		state = STATE_CHARSELECT
		results = cs_matches
	else:
		# Look for random button on stage selection
		t_b = dt()
		ss_matches = filterMatchTemplate(im_gray, tpl_stage, match_threshold)
		dt('mt stage', t_b)
		if(len(ss_matches) > 0):
			state = STATE_STAGESELECT
			results = ss_matches
		else:
			# Look for "0%" for in-game initialization
			t_c = dt()
			g_matches = filterMatchTemplate(im_gray, tpl_zero, match_threshold, True, 3)
			dt('mt zero', t_c)
			if(len(g_matches) > 1):
				state = STATE_INGAME
				results = g_matches

	dt('calibrate', tick)
	return state, results

def damageOCR(src_im, ROIs):
	tick = dt()

	gray_im = isolateChannel(src_im, 2)
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

		tick_mt = dt()
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
				digit_im = cv2.resize(digit_im, (10, 10))
				digit_1d = digit_im.reshape((1, 10 * 10))
				digit_1d = np.float32(digit_1d)
				digit, _res, _res, conf = model.find_nearest(digit_1d, k=1)

				roi_digits.insert(0, (int(digit), int(conf)))

			digits_str = ''.join([str(d[0]) for d in roi_digits])
			# digits_conf = ','.join([str(d[1]) for d in roi_digits])
			digits.append(digits_str)

			# cv2.putText(src_im, digits_str, loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# cv2.putText(src_im, '%0.2f' % match_val, (loc[0], loc[1] + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# cv2.putText(src_im, digits_conf, (loc[0], loc[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			
		else:
			digits.append('')

	dt('ocr', tick)
	return digits

def processVideo(video_path):
	video = cv2.VideoCapture(video_path)
	regions = []

	state = STATE_CALIBRATE
	prev_state = STATE_CALIBRATE
	events = []
	
	frames_without_digits = 0
	total_frames = video.get(7)
	prev_percent = 0

	while(video.isOpened()):
		tick = dt()

		ret, frame = video.read()
		if not ret: break

		frame_ms = video.get(0)
		frame_num = video.get(1)
		frame_progress = int((frame_num / total_frames) * 100)
		if(frame_progress != prev_percent):
			print '%d%%...' % prev_percent
		prev_percent = frame_progress

		if state != STATE_INGAME:
			state, regions = calibrateFrame(frame)

		# If calibration was unsuccessful, skip 5 frames
		if state == STATE_CALIBRATE:
			video.set(1, frame_num + 5)
			continue

		# Track state changes
		if prev_state != state:
			events.append((frame_ms, state, []))
			print 'State change: %d ms, %d' % (frame_ms, state)

		prev_state = state

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

		dt('frame', tick)

	video.release()
	print events
	#cv2.destroyAllWindows()
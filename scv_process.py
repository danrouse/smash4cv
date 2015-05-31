import scv_common as scv
import knearest

import cv2
import numpy as np

import os
import json
from glob import glob

import argparse

# TODO: Add support for processing multiple videos at once
parser = argparse.ArgumentParser()
parser.add_argument('--video_id', help='Basename of input video', default='mtZiCgiqHWU')
args = parser.parse_args()

kl_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kl_square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

tpl_zero = cv2.imread('%s/zero-percent-color-small.png' % scv.config['path']['templates'], 0)
tpl_percent = cv2.imread('%s/percent-sign.png' % scv.config['path']['templates'], 0)
tpl_csel = cv2.imread('%s/char-select.png' % scv.config['path']['templates'], 0)
tpl_ssel = cv2.imread('%s/stage-select.png' % scv.config['path']['templates'], 0)

knn_names = knearest.kNN('names')
knn_digits = knearest.kNN('digits')

class SmashVideo:
	def __init__(self, video_id, params = {}, debug_level = scv.DEBUG_NONE):
		scv.debug_level = debug_level

		# Video processing state
		self.state = scv.State.unknown	# Last detected state
		self.regions = []			# Matched ROIs to process
		self.template_scale = 1.0
		self.digit_cache = []
		self.death_cache = []
		self.cur_time = 0			# Video time in ms
		self.cur_frame = 0
		self.cur_progress = 0.0		# Position in video (0.0-1.0)
		self.events = [[] for _x in range(0, scv.State.quit.value)]	# Detected game events
		self.paused = False

		self.games = []
		self.cur_game = {
			'fighters': {},
			'events': [],
			'stage': '',
			'start': 0
		}

		self.video_id = video_id
		self.video = cv2.VideoCapture('%s/%s.mp4' % (scv.config['path']['videos'], video_id))
		self.total_frames = self.video.get(7)

	def process_video(self):
		if not self.video.isOpened():
			raise ValueError('Video is not loaded')

		while self.video.isOpened():
			ret, frame = self.video.read()
			if not ret: break

			# Update position counters
			self.cur_time = int(self.video.get(0))
			self.cur_frame = int(self.video.get(1))
			progress = int((self.cur_frame / self.total_frames) * 100)
			if progress != self.cur_progress:
				self.cur_progress = progress
				scv.log('Progress: %d%%' % progress)

			new_state = self.detect_state(frame)
			if self.state != new_state:
				scv.log(('State changed from %s to %s' % (self.state, new_state), self.cur_time), scv.DEBUG_EVENTS)
				self.state = new_state

			# Show video if debug flag set.
			# Navigate video with arrows, space pauses, Q quits
			if scv.debug_level & scv.DEBUG_VIDEO:
				if self.paused:
					scv.log(('Frame %d: %0.2fs' % (self.cur_frame, float(self.cur_time) / 1000),))

				cv2.imshow('Smash4CV', frame)

				key = cv2.waitKey(0 if self.paused else 1)
				if key == 113: # q(uit)
					break
				elif key == 32: # space (pause/unpause)
					self.paused = not self.paused
				elif key == 65361: # left arrow
					self.paused = True
					self.video.set(1, self.cur_frame - 2)
				elif key == 65363: # right arrow
					self.paused = True

		output_path = '%s/%s.json' % (scv.config['path']['output'], self.video_id)
		with open(output_path, 'w') as fp:
			if len(self.cur_game['events']) > 0:
				# flush cached game
				self.games.append(self.cur_game)
			json.dump(self.games, fp)
			scv.log('Saved JSON data to %s' % output_path)

	#@scv.benchmark
	def detect_state(self, src_im):
		state = scv.State.unknown

		im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
		im_h, im_w = im_gray.shape
		is_loading = self.detect_state_loading(im_gray)

		if is_loading:
			if self.state != scv.State.loading:
				# Entering loading state
				# Attempt to detect out-of-game state data from recent frame
				self.video.set(1, self.cur_frame - 21)
				ret, state_im = self.video.read()
				detected_state = self.detect_state_oog(state_im)

				# Reset video to original position
				self.video.set(1, self.cur_frame + 1)

			else:
				# In the middle of loading state
				pass

			state = scv.State.loading
		else:
			if self.state == scv.State.loading:
				# Exiting loading state
				# Attempt to detect an in-game state in coming frame
				self.video.set(1, self.cur_frame + 19)
				ret, state_im = self.video.read()
				regions, scale = self.detect_state_ig(state_im)
				if regions:
					# Initialize in-game state
					self.regions = regions
					self.template_scale = scale
					self.digit_cache = [0] * len(regions)
					self.death_cache = [0] * len(regions)

					state = scv.State.ingame

				# Reset video to original position
				self.video.set(1, self.cur_frame + 1)

			elif self.state == scv.State.ingame:
				# Continued in-game state
				for i,(x,y) in enumerate(self.regions):
					# Extract ROI
					x1 = int(x - ((im_w * self.template_scale)/13))
					x2 = int(x + ((im_w * self.template_scale)/11.9))
					y1 = int(y - ((im_h * self.template_scale)/51))
					y2 = int(y + ((im_h * self.template_scale)/8.5))
					region_im = src_im[y1:y2,x1:x2]

					region_im = cv2.resize(region_im, (100, 50))

					# Detect name if no good match yet
					if i not in self.cur_game['fighters']:
						name_im = region_im[40:50, 0:64]
						name_im = cv2.cvtColor(name_im, cv2.COLOR_BGR2GRAY)
						name, _res, name_conf = knn_names.identify(name_im)
						if name_conf <= 2000:
							scv.log(('Detected fighter', i, name), scv.DEBUG_DETECT)
							self.cur_game['fighters'][i] = name

					# Detect digits and compare
					digits_im = region_im[10:40, 30:100]
					digits = self.detect_digits(digits_im)
					#print(self.cur_game['fighters'][i] if i in self.cur_game['fighters'] else 'Unknown', digits)

					
					# Death must be detected continously for 15 frames to trigger
					if digits == scv.DETECT_DEAD:
						self.death_cache[i] -= 1
						if self.death_cache[i] > -15:
							digits = scv.DETECT_UNKNOWN

					# Add event when changed
					if digits > scv.DETECT_UNKNOWN and digits != self.digit_cache[i]:
						if digits != scv.DETECT_DEAD:
							self.death_cache[i] = 0
						self.digit_cache[i] = digits
						self.cur_game['events'].append((self.cur_time, i, digits))
						scv.log((self.cur_frame, self.cur_game['fighters'][i] if i in self.cur_game['fighters'] else 'Fighter %d' % i, '%d%%' % digits), scv.DEBUG_EVENTS)

					# Show ROIs while debugging
					if scv.debug_level & scv.DEBUG_VIDEO:
						cv2.circle(src_im, (x, y), 4, (0, 255, 0), 2)
						cv2.rectangle(src_im, (x1, y1), (x2, y2), (0, 0, 255), 1)
				
				state = scv.State.ingame

		return state

	def detect_state_loading(self, src_im):
		black_amount = self.count_black(src_im, 40)
		return (black_amount >= scv.config["threshold"]["loading_black"])

	@scv.benchmark
	def detect_state_oog(self, src_im):
		matches, conf, scale = self.match_template(src_im, tpl_ssel)
		if matches:
			scv.log(('Found stageselect template', matches, conf, scale), scv.DEBUG_DETECT)
			# TODO: Filter, find stage text contour, KNN
		else:
			# TODO: Possibly do something with character selection page?
			#	Can potentially extract player tags and costumes
			# matches, conf, scale = self.match_template(src_im, tpl_csel)
			# if matches:
			# 	scv.log(('Found charselect template', matches, conf, scale), scv.DEBUG_DETECT)
			pass

		return scv.State.unknown

	@scv.benchmark
	def detect_state_ig(self, src_im):
		# Find '0%' positions
		matches, conf, scale = self.match_template(src_im, tpl_zero)
		if len(matches) > 1:
			return matches, scale
		else:
			return [], 1.0

	@scv.benchmark
	def detect_digits(self, src_im, save_to_file=False):
		im_red = self.extract_channel(src_im, 2)

		# Find %-sign
		p_im = im_red[0:30, 35:70]	# search right half of image only
		p_match = cv2.matchTemplate(p_im, tpl_percent, cv2.TM_CCOEFF_NORMED)
		_res, p_max, _res, p_loc = cv2.minMaxLoc(p_match)

		digits = 0
		if p_max > 0.8:
			# Calculate (centered) number of digits by %-sign offset from center
			num_digits = int(p_loc[0] / 9) + 1

			for i in np.arange(num_digits, 0, -1):
				# TODO: Magic numbers
				digit_x = p_loc[0] - 33 - (19 * i)
				digit_im = im_red[3:29, digit_x:digit_x+18]
				
				digit, _res, digit_conf = knn_digits.identify(digit_im)
				digit = int(digit)

				# Inflate previous place value and add new digit
				digits = (digits * 10) + digit

				# Save for corrective training
				if save_to_file:
					cv2.imwrite('%s/digits/%d-%s-%d.png' % (scv.config['path']['training'], digit, self.video_id, self.cur_frame))

				# Give up on bad matches, we can catch it next frame
				if digit_conf > scv.config['knn']['digits']['conf']:
					digits = scv.DETECT_UNKNOWN
					break
		else:
			# Percent sign not found, try to detect death
			im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
			black_amount = self.count_black(im_gray, 150)
			#_,im_thresh=cv2.threshold(im_gray,120,255,cv2.THRESH_BINARY)
			#cv2.imshow('thresh', im_thresh)
			#print(black_amount)
			if black_amount > scv.config['threshold']['dead_black_high'] or black_amount < scv.config['threshold']['dead_black_low']:
				digits = scv.DETECT_DEAD
			else:
				digits = scv.DETECT_UNKNOWN
		
		return digits

	# @scv.benchmark
	# def process_region(self, src_im, save_digits=False):
	# 	# cvt red: 0.05ms
	# 	damage_im_color = src_im[10:40, 30:100]
	# 	damage_im = self.extract_channel(damage_im_color, 2)

	# 	# find percent: 0.25ms
	# 	#cv2.imshow('d%d' % x, damage_im)
	# 	p_match = cv2.matchTemplate(damage_im, tp_percent, cv2.TM_CCOEFF_NORMED)
	# 	_res, p_max, _res, p_loc = cv2.minMaxLoc(p_match)

	# 	digits = 0
	# 	if p_max > 0.8 and p_loc[0] > 35:
	# 		# digits: 0.6-1ms
	# 		num_digits = ((p_loc[0] - 35) / 10) + 1
	# 		#print num_digits

	# 		for i in np.arange(num_digits, 0, -1):
	# 			digit_x = p_loc[0] + 2 - (19 * i)
	# 			digit_im = damage_im[3:29, digit_x:digit_x+18]

	# 			digit, _res, digit_conf = knn_digits.identify(digit_im)
	# 			digits = (digits * 10) + digit

	# 			if save_digits:
	# 				cv2.imwrite('training/digits/%d-%s-%d.png' % (int(digit), self.video_id, self.cur_frame), digit_im)

	# 			if digit_conf > scv.config['knn']['digits']['conf']:
	# 				digits = scv.DETECT_UNKNOWN
	# 				break

	# 		if self.debug_level > 0:
	# 			cv2.circle(src_im, p_loc, 3, (0, 255, 255), 2)
	# 	else:
	# 		# detect death
	# 		damage_gray = cv2.cvtColor(damage_im_color, cv2.COLOR_BGR2GRAY)
	# 		damage_black = self.count_black(damage_gray, 150)

	# 		if damage_black > 0.95 or damage_black < 0.045:
	# 			digits = scv.DETECT_DEAD
	# 		else:
	# 			digits = scv.DETECT_UNKNOWN

	# 	return digits

	def count_black(self, src_im, threshold=127):
		"""
		Counts the amount of dark pixels in an image.
		
		Args:
			src_im (np array): source image
			threshold (int): darkness threshold (0 - 255)
		Returns:
			black_amount (float): percentage of dark pixels (0.0 - 1.0)
		"""

		_res, im_thresh = cv2.threshold(src_im, threshold, 255, cv2.THRESH_BINARY)
		nonzero_ct = cv2.countNonZero(im_thresh)
		return 1.0 - (float(nonzero_ct) / (src_im.shape[0] * src_im.shape[1]))

	def extract_channel(self, src_im, channel):
		im_ch = cv2.split(src_im)[channel]
		dst_im = src_im.copy()
		for i in range(0, src_im.shape[2]):
			dst_im[:,:,i] = im_ch
		dst_im = cv2.cvtColor(dst_im, cv2.COLOR_BGR2GRAY)
		return dst_im

	@scv.benchmark
	def match_template(self, src_im, tpl_im, threshold=0.8, max_scale=1.0, min_scale=0.25, scale_step=0.05):
		im_gray = src_im
		if len(src_im.shape) > 2:
			im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)

		best_scale = max_scale
		best_conf = 0
		best_coords = []

		im_h, im_w = im_gray.shape
		# Resize image in steps down and save the best overall match(es)
		for i in np.arange(max_scale, min_scale, -1 * scale_step):
			dest_shape = (int(im_w * i), int(im_h * i))
			if dest_shape[0] > tpl_im.shape[0] and dest_shape[1] > tpl_im.shape[1]:
				im_scaled = cv2.resize(im_gray, dest_shape)
				matches = cv2.matchTemplate(im_scaled, tpl_im, cv2.TM_CCOEFF_NORMED)
				results = np.where(matches > threshold)
				if len(results[0]) > 0:
					# filter results
					coords = np.int0(results[::-1] / i)	# scale size to base image
					rounded = np.int0(coords / 20) * 20 # round to nearest 20
					_res, unique = np.unique(rounded[0], True)	# group rounded values
					coords[1] = [min(coords[1])] * len(coords[1]) # reduce Y to minimum matched value
					coords = list(zip(coords[0][unique], coords[1][unique])) # grouped values only

					conf = np.sum(matches[results][unique])

					if conf > best_conf and len(coords) >= len(best_coords):
						best_conf = conf
						best_coords = coords
						best_scale = i
			else:
				# Too small to continue
				break

		return best_coords, best_conf, best_scale

#stats = SmashVideo('8uqAAppaCa4', debug_level = 5)
stats = SmashVideo(args.video_id, debug_level = scv.DEBUG_ALL ^ scv.DEBUG_PERF)
stats.process_video()
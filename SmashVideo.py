from glob import glob

import os
import cv2
import numpy as np
import youtube_dl
import json

with open('config.json') as fp:
	config = json.load(fp)

from kNN import kNN

STATE_UNKNOWN	= 1		
STATE_LOADING	= 2
STATE_INGAME	= 3
STATE_QUIT		= 4		# Used to quit in debug mode

DETECT_DEAD		= -1
DETECT_UNKNOWN	= -2

DEBUG_NONE		= 0b0000
DEBUG_NOTICE	= 0b0001
DEBUG_VIDEO		= 0b0010
DEBUG_EVENTS	= 0b0100
DEBUG_PERF		= 0b1000
DEBUG_ALL		= 0b1111
SHOW_BENCHMARK	= False

kl_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kl_square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

tpl_zero = cv2.imread('%s/zero-percent-color-small.png' % config['path']['templates'], 0)
tpl_percent = cv2.imread('%s/percent-sign.png' % config['path']['templates'], 0)
tpl_csel = cv2.imread('%s/char-select.png' % config['path']['templates'], 0)
tpl_ssel = cv2.imread('%s/stage-select.png' % config['path']['templates'], 0)

knn_names = kNN('names')
knn_digits = kNN('digits')

class SmashVideo:
	def __init__(self, yt_id, params = {}, debug_mode = DEBUG_NONE):
		self.debug_mode = debug_mode
		self.debug_log = []
		if debug_mode & DEBUG_PERF:
			globals()['SHOW_BENCHMARK'] = True

		# Video processing state
		self.state = STATE_UNKNOWN	# Last detected state
		self.regions = []			# Matched ROIs to process
		self.cache = {}				# In-state cache
		self.cur_time = 0			# Video time in ms
		self.cur_frame = 0
		self.cur_progress = 0.0		# Position in video (0.0-1.0)
		self.events = [[] for _x in range(0, STATE_QUIT)]	# Detected game events
		self.template_scale = 0.0	# Best scale from last scale independent match
		self.paused = False

		self.games = []
		self.cur_game = {
			'fighters': {},
			'events': [],
			'start': 0
		}

		self.video_id = yt_id
		self.video = self.load_video(yt_id)
		self.total_frames = self.video.get(7)
		#self.process_video()

	def log(self, data, log_level = DEBUG_NOTICE):
		self.debug_log.append((log_level, data))
		if self.debug_mode & log_level:
			if type(data) is str:
				print(data)
			else:
				print(': '.join([str(d) for d in data]))

	def benchmark(method):
		def timer(*args, **kw):
			if not SHOW_BENCHMARK:
				return method(*args, **kw)

			tickA = cv2.getTickCount()
			method(*args, **kw)
			tickB = cv2.getTickCount()
			diff = (tickB - tickA) / cv2.getTickFrequency()
			print(': '.join((method.__name__, '%.3fms' % (diff * 1000))))

		return timer

	def load_video(self, yt_id):
		if not os.path.exists('%s/%s.mp4' % (config['path']['videos'], yt_id)):
			self.log(('Downloading video', yt_id), DEBUG_NOTICE)
			yt_opts = dict(
				format = '134/135/mp4[acodec=none]',
				outtmpl = '%s/%s.mp4' % (config['path']['videos'], '%(id)s'))
			with youtube_dl.YoutubeDL(yt_opts) as ytdl:
				ytdl.download(['https://www.youtube.com/watch?v=%s' % yt_id])

		return cv2.VideoCapture('%s/%s.mp4' % (config['path']['videos'], yt_id))

	@benchmark
	def process_video(self):
		if not self.video.isOpened():
			raise ValueError('Video is not loaded')

		while self.video.isOpened():
			ret, frame = self.video.read()
			if not ret: break

			self.cur_time = int(self.video.get(0))
			self.cur_frame = self.video.get(1)

			progress = int((self.cur_frame / self.total_frames) * 100)
			if progress != self.cur_progress:
				self.cur_progress = progress
				self.log('Progress: %d%%' % progress, DEBUG_NOTICE)

			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			is_loading = self.detect_loading(frame_gray)

			if is_loading and self.state != STATE_LOADING:
				# Entering loading state
				# Attempt to detect out-of-game state
				self.video.set(1, self.cur_frame - 21)
				ret, state_frame = self.video.read()
				detected_state = self.detect_state_oog(state_frame)

				self.video.set(1, self.cur_frame + 1)
				self.state = STATE_LOADING
			elif not is_loading and self.state == STATE_LOADING:
				# Exiting loading state
				# Attempt to detect in-game state
				self.video.set(1, self.cur_frame + 19)
				ret, state_frame = self.video.read()
				detected_state = self.detect_state_ig(state_frame)

				self.video.set(1, self.cur_frame + 1)
				self.state = STATE_UNKNOWN
			elif not is_loading and self.state != STATE_LOADING:
				# Neutral state
				pass

			# Show video if debug flag set.
			# Navigate video with arrows, space pauses, Q quits
			if self.debug_mode & DEBUG_VIDEO:
				if self.paused:
					self.log(('Frame %d: %0.2fs' % (self.cur_frame, float(self.cur_time) / 1000),), DEBUG_NOTICE)

				cv2.imshow('Smash Frame', frame)

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

		output_path = '%s/%s.json' % (config['path']['output'], self.video_id)
		with open(output_path, 'w') as fp:
			if len(self.cur_game['events']) > 0:
				self.games.append(self.cur_game)
			json.dump(self.games, fp)
			self.log(('Saved data', output_path), DEBUG_NOTICE)

	@benchmark
	def detect_loading(self, src_im):
		black_amount = self.count_black(src_im, 40)
		return (black_amount >= config["threshold"]["loading_black"])
		
		"""
		# iterate through ROIs
		for i,(x,y) in enumerate(self.regions):
			region_im = src_im[
				y-((im_h * self.template_scale)/51):y+((im_h * self.template_scale)/8.5),
				x-((im_w * self.template_scale)/13):x+((im_w * self.template_scale)/11.9)]
			region_im = cv2.resize(region_im, (100, 50))

			# name: 0.1ms
			if i not in self.cur_game['fighters']:
				name_im = region_im[40:50, 0:64]
				name_im = cv2.cvtColor(name_im, cv2.COLOR_BGR2GRAY)
				name, _res, name_conf = knn_names.identify(name_im)
				if name_conf <= 2000:
					self.cur_game['fighters'][i] = name

			# digit detection
			digits = self.process_region(region_im)

			# check death continuity
			if digits == DETECT_DEAD:
				if not self.cache[i]: self.cache[i] = 0
				self.cache[i] -= 1
				if self.cache[i] > -5:
					digits = DETECT_UNKNOWN
				elif (i not in self.cache or (i in self.cache and digits != self.cache[i])):
					self.cur_game['events'].append((self.cur_time, i, digits))
					self.log((self.cur_frame, i, digits), 1)
			elif digits > DETECT_UNKNOWN and (i not in self.cache or (i in self.cache and digits != self.cache[i])):
				self.cache[i] = digits
				self.cur_game['events'].append((self.cur_time, i, digits))
				self.log((self.cur_frame, i, digits), 1)
		"""

	@benchmark
	def detect_state_oog(self, src_im):
		matches, conf, scale = self.match_template(src_im, tpl_ssel)
		if matches:
			print(('stage', matches, conf))
		else:
			matches, conf, scale = self.match_template(src_im, tpl_csel)
			if matches:
				print(('char', matches, conf))
		return STATE_UNKNOWN

	@benchmark
	def detect_state_ig(self, src_im):
		matches, conf, scale = self.match_template(src_im, tpl_zero)
		for (x, y) in matches:
			print('match at %d,%d' % (x,y))
		cv2.imshow('detect-state-ig', src_im)

	@benchmark
	def process_region(self, src_im, save_digits=False):
		# cvt red: 0.05ms
		damage_im_color = src_im[10:40, 30:100]
		damage_im = self.extract_channel(damage_im_color, 2)

		# find percent: 0.25ms
		#cv2.imshow('d%d' % x, damage_im)
		p_match = cv2.matchTemplate(damage_im, tp_percent, cv2.TM_CCOEFF_NORMED)
		_res, p_max, _res, p_loc = cv2.minMaxLoc(p_match)

		digits = 0
		if p_max > 0.8 and p_loc[0] > 35:
			# digits: 0.6-1ms
			num_digits = ((p_loc[0] - 35) / 10) + 1
			#print num_digits

			for i in np.arange(num_digits, 0, -1):
				digit_x = p_loc[0] + 2 - (19 * i)
				digit_im = damage_im[3:29, digit_x:digit_x+18]

				digit, _res, digit_conf = knn_digits.identify(digit_im)
				digits = (digits * 10) + digit

				if save_digits:
					cv2.imwrite('training/digits/%d-%s-%d.png' % (int(digit), self.video_id, self.cur_frame), digit_im)

				if digit_conf > config['knn']['digits']['conf']:
					digits = DETECT_UNKNOWN
					break

			if self.debug_mode > 0:
				cv2.circle(src_im, p_loc, 3, (0, 255, 255), 2)
		else:
			# detect death
			damage_gray = cv2.cvtColor(damage_im_color, cv2.COLOR_BGR2GRAY)
			damage_black = self.count_black(damage_gray, 150)

			if damage_black > 0.95 or damage_black < 0.045:
				digits = DETECT_DEAD
			else:
				digits = DETECT_UNKNOWN

		return digits

	@benchmark
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

	@benchmark
	def match_template(self, src_im, tpl_im, threshold=0.8, max_scale=1.0, min_scale=0.25, scale_step=0.05):
		if len(src_im.shape) > 2:
			src_im = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)

		best_scale = max_scale
		best_match = 0
		best_match_coords = []

		im_h, im_w = src_im.shape
		for i in np.arange(max_scale, min_scale, -1 * scale_step):
			dest_shape = (int(im_w * i), int(im_h * i))
			if dest_shape[0] > tpl_im.shape[0] and dest_shape[1] > tpl_im.shape[1]:

				im_scaled = cv2.resize(src_im, dest_shape)
				cv2.imshow('pre',im_scaled)
				cv2.imshow('t', tpl_im)
				#cv2.waitKey(0)
				matches = cv2.matchTemplate(im_scaled, tpl_im, cv2.TM_CCOEFF_NORMED)
				results = np.where(matches > threshold)
				if len(results[0]) > 0:
					# filter results
					coords = np.int0(results[::-1] / i)
					rounded = np.int0(coords / 20) * 20 # group by nearest 20
					_res, unique = np.unique(rounded[0], True)
					conf = np.sum(matches[results][unique])
					coords[1] = [min(coords[1])] * len(coords[1]) # reduce Y to min value of match
					coords = list(zip(coords[0][unique], coords[1][unique]))

					if conf > best_match and len(coords) >= len(best_match_coords):
						best_match = conf
						best_match_coords = coords
						best_scale = i

		return best_match_coords, best_match, best_scale

#stats = SmashVideo('8uqAAppaCa4', debug_mode = 5)
stats = SmashVideo('mtZiCgiqHWU', debug_mode = DEBUG_VIDEO | DEBUG_EVENTS | DEBUG_NOTICE)
stats.process_video()
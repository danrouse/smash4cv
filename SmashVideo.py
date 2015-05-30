from __future__ import unicode_literals
from glob import glob

import os
import cv2
import numpy as np
import youtube_dl
import json

with open('config.json') as fp:
	config = json.load(fp)

from kNN import kNN

STATE_QUIT		= 0
STATE_UNKNOWN	= 1
STATE_LOADING	= 2

DETECT_DEAD		= -1
DETECT_UNKNOWN	= -2

kl_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kl_square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

tp_zero = cv2.imread('%s/zero-percent-color-small.png' % config['path']['templates'], 0)
tp_percent = cv2.imread('%s/percent-sign.png' % config['path']['templates'], 0)

# ocr = tesseract.TessBaseAPI()
# ocr.Init('.', 'eng', tesseract.OEM_DEFAULT)

class SmashVideo:
	knn_names = kNN('names')
	knn_digits = kNN('digits')

	def __init__(self, yt_id, params = {}, debug_mode = 0):
		self.debug_mode = debug_mode
		self.debug_log = []
		self.timers = {}

		self.state = STATE_UNKNOWN
		self.events = []
		self.regions = []
		self.cache = {}
		self.cur_time = 0
		self.cur_frame = 0
		self.cur_progress = 0.0
		self.template_scale = 0.0
		self.paused = False

		self.video_id = yt_id

		self.video = self.load_video(yt_id)
		self.total_frames = self.video.get(7)

		self.process_video()

	def log(self, data, log_level = 0):
		if type(data) is str:
			data = (data, )
		self.debug_log.append((log_level, data))
		if self.debug_mode >= log_level:
			data = [str(d) for d in data]
			print ': '.join(data)

	def benchmark(self, caller='generic', start=False, subtract=0):
		if self.debug_mode < 4: return
		
		tick = cv2.getTickCount()
		if caller in self.timers and not start:
			diff = (tick - self.timers[caller] - subtract) / cv2.getTickFrequency()
			del self.timers[caller]
			self.log((caller, '%.3fms' % (diff * 1000)), 4)
			return diff
		else:
			self.timers[caller] = tick
			return tick

	def load_video(self, yt_id):
		if not os.path.exists('%s/%s.mp4' % (config['path']['videos'], yt_id)):
			self.log(('Downloading video', yt_id))
			yt_opts = dict(
				format = '134/135/mp4[acodec=none]',
				outtmpl = '%s/%s.mp4' % (config['path']['videos'], '%(id)s'))
			with youtube_dl.YoutubeDL(yt_opts) as ytdl:
				ytdl.download(['https://www.youtube.com/watch?v=%s' % yt_id])

		return cv2.VideoCapture('%s/%s.mp4' % (config['path']['videos'], yt_id))

	def process_video(self):
		if not self.video.isOpened():
			raise ValueError('Video is not loaded')

		#self.video.set(1, 4600)
		while self.video.isOpened():
			ret, frame = self.video.read()
			if not ret: break

			self.cur_time = int(self.video.get(0))
			self.cur_frame = self.video.get(1)

			progress = int((self.cur_frame / self.total_frames) * 100)
			if progress != self.cur_progress:
				self.cur_progress = progress
				self.log(('Progress', '%d%%' % progress), 0)

			self.benchmark('Frame', True)
			self.state = self.process_frame(frame)
			self.benchmark('Frame')
			if self.state == STATE_QUIT: break

		output_path = '%s/%s.json' % (config['path']['output'], self.video_id)
		with open(output_path, 'w') as fp:
			json.dump(self.events, fp)
			self.log(('Saved data', output_path))

	def process_frame(self, src_im):
		im_h, im_w, _ = src_im.shape

		#im_redchannel = cv2.split(src_im)[2]
		im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)

		# black count: 0.15ms
		black_amount = self.count_black(im_gray, 40)
		if black_amount >= 0.75:
			#self.regions = []
			self.cache = {}
			return STATE_LOADING

		# get best scale
		if len(self.regions) < 2:
			self.benchmark('Find regions', True)
			best_scale = 1.0
			best_match = 0
			best_match_coords = []

			for i in np.arange(1.0 if self.template_scale == 0.0 else self.template_scale,
					           0.25 if self.template_scale == 0.0 else self.template_scale, -0.05):
				dest_shape = (int(im_w * i), int(im_h * i))
				if dest_shape[0] >= tp_zero.shape[0] * 10 and dest_shape[1] >= tp_zero.shape[1] * 5:
					im_scaled = cv2.resize(im_gray, dest_shape)
					matches = cv2.matchTemplate(im_scaled, tp_zero, cv2.TM_CCOEFF_NORMED)
					results = np.where(matches > 0.8)
					if len(results[0]) > 0:
						coords = np.int0(results[::-1] / i)
						rounded = np.int0(coords / 20) * 20
						_res, unique = np.unique(rounded[0], True)
						conf = np.sum(matches[results][unique])
						coords[1] = [min(coords[1])] * len(coords[1])
						coords = zip(coords[0][unique], coords[1][unique])
						if conf > best_match and len(coords) >= len(best_match_coords):
							best_match = conf
							best_match_coords = coords
							best_scale = i

			if len(best_match_coords) >= 2:
				self.template_scale = best_scale
				self.regions = best_match_coords
			self.benchmark('Find regions')

		# iterate through ROIs
		for x,y in self.regions:
			self.benchmark('Process ROI', True)
			player_im = src_im[
				y-(im_h/60):y+(im_h/10),
				x-(im_w/15):x+(im_w/14)]

			# resize: 0.05ms
			player_im = cv2.resize(player_im, (100, 50))
			#cv2.imshow('player%d' %x, player_im)
			
			# name: 0.1ms
			if x not in self.cache or (x in self.cache and self.cache[x][1] == 'Unknown'):
				name_im = player_im[40:50, 0:64]
				name_im = cv2.cvtColor(name_im, cv2.COLOR_BGR2GRAY)
				#cv2.imshow('nme', name_im)
				name, _res, name_conf = self.knn_names.identify(name_im)
				if name_conf > 2000:
					name = 'Unknown'
			else:
				name = self.cache[x][1]

			# cvt red: 0.05ms
			damage_im_color = player_im[10:40, 30:100]
			damage_im = self.extract_channel(damage_im_color, 2)

			# find percent: 0.25ms
			#cv2.imshow('d%d' % x, damage_im)
			p_match = cv2.matchTemplate(damage_im, tp_percent, cv2.TM_CCOEFF_NORMED)
			_res, p_max, _res, p_loc = cv2.minMaxLoc(p_match)

			digits = 0
			if p_max > 0.8 and p_loc[0] > 35:
				self.benchmark('Digit OCR', True)
				# digits: 0.6-1ms
				num_digits = ((p_loc[0] - 35) / 10) + 1
				#print num_digits

				for i in np.arange(num_digits, 0, -1):
					digit_x = p_loc[0] + 2 - (19 * i)
					digit_im = damage_im[3:29, digit_x:digit_x+18]

					digit, _res, digit_conf = self.knn_digits.identify(digit_im)
					digits = (digits * 10) + digit

					#cv2.imshow('digit-%d %d' % (x, i), digit_im)
					#if np.random.rand() > 0.5:
					#	cv2.imwrite('training/digits/%d-%d.png' % (int(digit), self.cur_frame), digit_im)

					if digit_conf > config['knn']['digits']['conf']:
						digits = DETECT_UNKNOWN
						break

				if self.debug_mode > 0:
					cv2.circle(src_im, (x+p_loc[0]-(im_w/30), y+p_loc[1]), 3, (0, 255, 255), 2)

				self.benchmark('Digit OCR')
			else:
				# detect death
				damage_gray = cv2.cvtColor(damage_im_color, cv2.COLOR_BGR2GRAY)
				damage_black = self.count_black(damage_gray, 150)

				if damage_black > 0.95 or damage_black < 0.045:
					digits = DETECT_DEAD
				elif x in self.cache:
					digits = self.cache[x][2]
				else:
					digits = DETECT_UNKNOWN

			# check death continuity
			cache_key = '%d-death' % x
			if digits == DETECT_DEAD:
				if cache_key not in self.cache:
					self.cache[cache_key] = 0

				self.cache[cache_key] += 1
				
				if self.cache[cache_key] < 5:
					digits = DETECT_UNKNOWN
			elif digits != DETECT_UNKNOWN:
				self.cache[cache_key] = 0

			if digits > DETECT_UNKNOWN and (x not in self.cache or (x in self.cache and self.cache[x][2] != digits)):
				
				self.events.append((self.cur_time, name, digits))
				self.log((self.cur_frame, name, digits), 1)

			if digits > DETECT_UNKNOWN:
				self.cache[x] = (self.cur_time, name, digits)
			
			self.benchmark('Process ROI')

		if self.debug_mode > 1:
			if self.paused:
				self.log(('Frame %d: %0.2fs' % (self.cur_frame, float(self.cur_time) / 1000),), 1)

			cv2.imshow('Smash Frame', src_im)

			key = cv2.waitKey(0 if self.paused else 1)
			if key == 113: # q(uit)
				return STATE_QUIT
			elif key == 32: # space (pause/unpause)
				self.paused = not self.paused
			elif key == 2424832: # left arrow
				self.paused = True
				self.video.set(1, self.cur_frame - 2)
			elif key == 2555904: # right arrow
				self.paused = True

		return STATE_UNKNOWN

	def add_event(self, event_type, event_data, event_time = 0):
		if event_time == 0:
			event_time = self.cur_time
		if event_type not in self.events:
			self.events[event_type] = []
		self.events[event_type].append((event_time, event_data))
		return self.events[event_type]

	def count_black(self, src_im, threshold=127):
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

#stats = SmashVideo('8uqAAppaCa4', debug_mode = 5)
stats = SmashVideo('3tai70AIoe4', debug_mode = 5)
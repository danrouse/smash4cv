#!/usr/bin/python3

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
#parser.add_argument('--video_id', help='Basename of input video', default='HEu4P2ope2I')
parser.add_argument('--video_id', help='Basename of input video', default='8uqAAppaCa4')

tpl_zero = cv2.imread('%s/zero-percent-color-small.png' % scv.config['path']['templates'], 0)
tpl_percent = cv2.imread('%s/percent-sign.png' % scv.config['path']['templates'], 0)
tpl_csel = cv2.imread('%s/char-select.png' % scv.config['path']['templates'], 0)
tpl_ssel = cv2.imread('%s/stage-select.png' % scv.config['path']['templates'], 0)

knn_names = knearest.kNN('names')
knn_digits = knearest.kNN('digits')
knn_stages = knearest.kNN('stages')

class SmashGame:
	def __init__(self, start=0):
		self.fighters = {}
		self.hits = []
		self.deaths = []
		self.combos = []
		self.stage = 'Unknown'
		self.start = start
		self.winner = -1

class SmashVideo:
	def __init__(self, video_id, params = {}, debug_level = scv.DEBUG_NOTICE):
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
		self.paused = False

		self.games = []
		self.cur_game = SmashGame()

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

	@scv.benchmark
	def detect_state(self, src_im):
		state = scv.State.unknown

		unknown_regions = 0
		if self.state == scv.State.ingame:
			# Continued in-game state
			for i,(x,y) in enumerate(self.regions):
				# Extract ROI
				x1 = int(x - (self.template_scale * 73))
				x2 = int(x + (self.template_scale * 75))
				y1 = int(y - (self.template_scale * 10))
				y2 = int(y + (self.template_scale * 61))
				region_im = src_im[y1:y2,x1:x2]

				region_im = cv2.resize(region_im, (100, 50))

				# Detect name if no good match yet
				if i not in self.cur_game.fighters:
					name_im = region_im[40:50, 0:67]
					name_im = cv2.cvtColor(name_im, cv2.COLOR_BGR2GRAY)
					cv2.imshow('name', name_im)

					name, _res, name_conf = knn_names.identify(name_im)
					if name_conf <= 2000:
						scv.log(('Detected fighter', i, name), scv.DEBUG_DETECT)
						self.cur_game.fighters[i] = name

				# Detect digits and compare
				digits_im = region_im[10:40, 30:100]
				digits = self.read_digits(digits_im)
				
				# Death must be detected continously for 15 frames to trigger
				if digits == scv.DETECT_DEAD:
					self.death_cache[i] -= 1
					if self.death_cache[i] > -15:
						digits = scv.DETECT_UNKNOWN

				# Add event when changed
				if digits > scv.DETECT_UNKNOWN and digits != self.digit_cache[i]:
					if digits != scv.DETECT_DEAD:
						self.death_cache[i] = 0
						self.cur_game.hits.append((self.cur_time, i, digits))
						scv.log((self.cur_frame, self.cur_game.fighters[i] if i in self.cur_game.fighters else 'Fighter %d/%d' % (i, len(self.regions)), '%d%%' % digits), scv.DEBUG_EVENTS)
					else:
						self.cur_game.deaths.append((self.cur_time, i))
						scv.log((self.cur_frame, self.cur_game.fighters[i] if i in self.cur_game.fighters else 'Fighter %d/%d' % (i, len(self.regions)), 'died'), scv.DEBUG_EVENTS)
					self.digit_cache[i] = digits
				
				if digits == scv.DETECT_UNKNOWN:
					unknown_regions += 1

				# Show ROIs while debugging
				if scv.debug_level & scv.DEBUG_VIDEO:
					cv2.circle(src_im, (x, y), 4, (0, 255, 0), 2)
					cv2.rectangle(src_im, (x1, y1), (x2, y2), (0, 0, 255), 1)
			
			state = scv.State.ingame

		if state != scv.State.ingame or unknown_regions == len(self.regions):
			im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
			im_h, im_w = im_gray.shape
			is_loading = self.detect_state_loading(im_gray)

			if is_loading:
				if self.state != scv.State.loading:
					# Entering loading state
					if self.state == scv.State.ingame:
						# Save any previously detected game
						if len(self.cur_game.hits) > 0:
							self.games.append(self.cur_game)

						# Reset game state
						self.cur_game = SmashGame(start=self.cur_time)

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
			elif self.state == scv.State.loading or self.state == scv.State.unknown:
				# Exiting loading state
				# Attempt to detect an in-game state in coming frame
				self.video.set(1, self.cur_frame + 29)
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

		return state

	def detect_state_loading(self, src_im):
		black_amount = self.count_black(src_im, 20)
		return (black_amount >= scv.config["threshold"]["loading_black"])

	@scv.benchmark
	def detect_state_oog(self, src_im):
		im_gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
		matches, conf, scale = self.match_template(im_gray, tpl_ssel, threshold=0.7)
		if matches:
			im_h, im_w = im_gray.shape
			im_cropped = im_gray[0:im_h, 0:int(im_w/2)]
			stage, _res, conf = knn_stages.identify(im_cropped)

			self.cur_game.stage = stage
			scv.log((self.cur_frame, 'Entering stage', stage), scv.DEBUG_EVENTS)
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
		return matches, scale

	#@scv.benchmark
	def read_digits(self, src_im, save_to_file=False):
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

			if black_amount > scv.config['threshold']['dead_black_high'] or black_amount < scv.config['threshold']['dead_black_low']:
				digits = scv.DETECT_DEAD
			else:
				digits = scv.DETECT_UNKNOWN
		
		return digits

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
	def match_template(self, src_im, tpl_im, threshold=0.8, max_scale=1.0, min_scale=0.25, scale_step=0.025):
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
					rounded = np.round(coords / 20) # round to nearest 20
					_res, unique = np.unique(rounded[0], True)	# group rounded values
					coords[1] = [min(coords[1])] * len(coords[1]) # reduce Y to minimum matched value
					coords = list(zip(coords[0][unique], coords[1][unique])) # grouped values only

					conf = np.mean(matches[results][unique])

					#print(coords, rounded, unique, conf)

					if conf > best_conf:
						best_conf = conf
						best_coords = coords
						best_scale = i
			else:
				# Too small to continue
				break

		return best_coords, best_conf, best_scale

	def process_game_data(self):
		for game in self.games:
			# last person dead is the winner:
			#   they get detected as dead when the game ends, after their defeated foe.
			if len(game.deaths) > 0:
				game.winner = game.deaths.pop()

			# combos: N hits in a row
			combo_counter = 0
			combo_victim = -1
			combo_start_time = 0
			combo_start_hp = 0
			combo_last_hp = 0
			for hit in game.hits:
				# hit(time, victim, digits)
				if hit[1] != combo_victim:
					# record combos above threshold
					if combo_counter >= scv.config['threshold']['hits_for_combo']:
						game.combos.append((combo_start_time, hit[0], combo_victim, combo_counter, combo_start_hp, combo_last_hp))

					# reset combo
					combo_counter = 0
					combo_victim = hit[1]
					combo_start_time = hit[0]
					combo_start_hp = hit[2]
				else:
					combo_counter += 1
					combo_last_hp = hit[2]

	def save_highlights(self):
		i = 0
		fps = int(self.video.get(5))
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		size = (int(self.video.get(3)), int(self.video.get(4)))
		videos = []

		for game in self.games:
			# save kills
			for death in game.deaths:
				patient = death[1]
				agent = int(not patient) # this won't support more than two players

				i += 1
				video_path = '%s/ko-%s-%s-%s-%d.avi' % (
					scv.config['path']['output'],
					game.fighters[agent],
					game.fighters[patient], self.video_id, i)
				videos.append((video_path, death[0] - 5000, death[0] + 1000))

			# save combos
			for combo in game.combos:
				patient = combo[2]
				agent = int(not patient) # this won't support more than two players

				i += 1
				video_path = '%s/combo-%s-%s-%s-%d.avi' % (
					scv.config['path']['output'],
					game.fighters[agent],
					game.fighters[patient], self.video_id, i)
				videos.append((video_path, combo[0] - 1000, combo[1] + 250))

			# save videos
			for i,video in enumerate(videos):
				scv.log(('Saving video', i, video[0]))
				writer = cv2.VideoWriter(video[0], fourcc, fps, size)
				self.video.set(0, video[1])
				while self.video.get(0) < video[2]:
					ret, frame = self.video.read()
					writer.write(frame)
				writer.release()

	def save(self):
		# flush cached game
		if len(self.cur_game.hits) > 0:
			self.games.append(self.cur_game)

		self.process_game_data()
		self.save_highlights()
		output_path = '%s/%s.json' % (scv.config['path']['output'], self.video_id)
		with open(output_path, 'w') as fp:
			
			json.dump([game.__dict__ for game in self.games], fp)
			scv.log('Saved JSON data to %s' % output_path)

if __name__ == '__main__':
	args = parser.parse_args()
	stats = SmashVideo(args.video_id)
	stats.process_video()
	stats.save()
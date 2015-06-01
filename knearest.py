#!/usr/bin/python3

import scv_common as scv

import cv2
import numpy as np
from glob import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='model', help='Model to train', nargs='+', default='none')
parser.add_argument('--list', help='List available models', action='store_true')

class kNN:
	def __init__(self, name, training=False):
		self.model = cv2.ml.KNearest_create()
		self.name = name
		self.token_list = []

		if name in scv.config['knn']:
			self.tokens = scv.config['knn'][name]['tokens']
			self.size = tuple(scv.config['knn'][name]['size'])
			self.threshold = scv.config['knn'][name]['threshold']
			self.find_contours = scv.config['knn'][name]['contours']
		else:
			raise ValueError('Model not found')

		if training:
			self.load_sources()
		else:
			sdata = np.loadtxt('%s/%s-samples.data' % (scv.config['path']['training'], name), np.float32)
			rdata = np.loadtxt('%s/%s-responses.data' % (scv.config['path']['training'], name), np.float32)
			rdata = rdata.reshape((rdata.size, 1))
			self.model.train(sdata, cv2.ml.ROW_SAMPLE, rdata)

			if self.tokens:
				with open('%s/%s-tokens.data' % (scv.config['path']['training'], name)) as f:
					tdata = [line.rsplit('\n') for line in f.readlines()]
					self.token_list = tdata

	def identify(self, source, k=1):
		if self.find_contours:
			source = self.largest_bbox(source)

		source = cv2.resize(source, self.size)

		if self.threshold > 0:
			_res, source = cv2.threshold(source, self.threshold, 255, cv2.THRESH_BINARY)

		matrix = source.reshape((1, self.size[0] * self.size[1]))
		matrix = np.float32(matrix)
		match, _, matches, confs = self.model.findNearest(matrix, k=k)
		confs /= (self.size[0] * self.size[1])
		confs /= k
		confs = confs[0]
		if self.tokens:
			matched_tokens = [self.token_list[int(i)][0] for i in matches[0]]

			return self.token_list[int(match)][0], matched_tokens, confs
		else:
			return int(match), matches, confs

	def train(self, sources=None, responses=None):
		if sources == None:
			sources = self.sources
		if responses == None:
			responses = self.responses
		samples = np.empty((0, self.size[0] * self.size[1]))
		for i, source in enumerate(sources):
			if self.find_contours:
				source = self.largest_bbox(source)

			source = cv2.resize(source, (self.size[0], self.size[1]))

			if self.threshold > 0:
				_res, source = cv2.threshold(source, self.threshold, 255, cv2.THRESH_BINARY)			

			sample = source.reshape((1, self.size[0] * self.size[1]))
			samples = np.append(samples, sample, 0)

		if self.tokens:
			np.savetxt('%s/%s-tokens.data' % (scv.config['path']['training'], self.name), responses, '%s')
			responses = range(0, len(responses))
			print('Wrote tokens to %s/%s-tokens.data' % (scv.config['path']['training'], self.name))

		responses = np.array(responses, np.float32)
		responses = responses.reshape((responses.size, 1))
		np.savetxt('%s/%s-samples.data' % (scv.config['path']['training'], self.name), samples, '%d')
		np.savetxt('%s/%s-responses.data' % (scv.config['path']['training'], self.name), responses, '%d')

		print('Wrote samples to %s/%s-samples.data' % (scv.config['path']['training'], self.name))
		print('Wrote responses to %s/%s-responses.data' % (scv.config['path']['training'], self.name))


	def load_sources(self, ext='.png'):
		path = '%s/%s/*%s' % (scv.config['path']['training'], self.name, ext)
		sources = []
		responses = []
		for f in glob(path):
			tpl = cv2.imread(f, 0)
			name = f.replace('\\', '/').split('/')[-1].split('-')[0].split(' ')[0].replace('_', ' ').replace('.png', '')
			#print name
			sources.append(tpl)
			responses.append(name)

		self.sources = sources
		self.responses = responses

	def largest_bbox(self, src_im):
		_res, im_thresh = cv2.threshold(src_im, 240, 255, cv2.THRESH_BINARY)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
		im_morph = cv2.dilate(im_thresh, kernel, 2)
		_res, contours, _res = cv2.findContours(im_morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		largest_area = 0
		largest_bbox = []
		largest_contour = []
		for c in contours:
			area = cv2.contourArea(c)
			if area > largest_area:
				largest_area = area
				largest_bbox = cv2.boundingRect(c)
				largest_contour = c
		return src_im[
			largest_bbox[1]:largest_bbox[1]+largest_bbox[3],
			largest_bbox[0]:largest_bbox[0]+largest_bbox[2]]

if __name__ == '__main__':
	args = parser.parse_args()
	if args.list:
		print('Available models:')
		for i in scv.config['knn'].keys():
			print('\t%s' % i)
	else:
		for i in args.model:
			if i == 'all':
				for j in scv.config['knn'].keys():
					model = kNN(j, training=True)
					model.train()
			else:
				model = kNN(i, training=True)
				model.train()
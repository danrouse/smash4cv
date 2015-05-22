import cv2
import numpy as np
import json
from glob import glob

with open('config.json') as fp:
	config = json.load(fp)

class kNN:

	def __init__(self, name, training=False, size=(10,10), tokens=True, threshold=0):
		self.model = cv2.KNearest()
		self.name = name
		self.token_list = []

		if name in config['knn']:
			self.tokens = config['knn'][name]['tokens']
			self.size = tuple(config['knn'][name]['size'])
			self.threshold = config['knn'][name]['threshold']
		else:
			self.tokens = tokens
			self.size = size
			self.threshold = threshold

		if training:
			self.load_sources()
		else:
			sdata = np.loadtxt('%s/%s-samples.data' % (config['path']['training'], name), np.float32)
			rdata = np.loadtxt('%s/%s-responses.data' % (config['path']['training'], name), np.float32)
			rdata = rdata.reshape((rdata.size, 1))
			self.model.train(sdata, rdata)

			with open('%s/%s-tokens.data' % (config['path']['training'], name)) as f:
				tdata = [line.rsplit('\n') for line in f.readlines()]
				self.token_list = tdata

	def identify(self, source, k=1):
		if self.threshold > 0:
			_res, source = cv2.threshold(source, self.threshold, 255, cv2.THRESH_BINARY)
		source = cv2.resize(source, self.size)

		# cv2.imshow('ident', source)
		# cv2.waitKey(1)

		matrix = source.reshape((1, self.size[0] * self.size[1]))
		matrix = np.float32(matrix)
		match, _, matches, confs = self.model.find_nearest(matrix, k=k)
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
			source = cv2.resize(source, (self.size[0], self.size[1]))
			if self.threshold > 0:
				_res, source = cv2.threshold(source, self.threshold, 255, cv2.THRESH_BINARY)

			sample = source.reshape((1, self.size[0] * self.size[1]))
			samples = np.append(samples, sample, 0)

		if self.tokens:
			np.savetxt('%s/%s-tokens.data' % (config['path']['training'], self.name), responses, '%s')
			responses = range(0, len(responses))

		responses = np.array(responses, np.float32)
		responses = responses.reshape((responses.size, 1))
		np.savetxt('%s/%s-samples.data' % (config['path']['training'], self.name), samples, '%d')
		np.savetxt('%s/%s-responses.data' % (config['path']['training'], self.name), responses, '%d')


	def load_sources(self, ext='.png'):
		path = '%s/%s/*%s' % (config['path']['training'], self.name, ext)
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

if __name__ == '__main__':
	knn_names = kNN('names', training=True)
	knn_names.train()

	knn_digits = kNN('digits', training=True)
	knn_digits.train()
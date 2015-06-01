#!/usr/bin/python3

import scv_common as scv

import youtube_dl
import os.path

import argparse

# TODO: Add support for YouTube channels with filter
parser = argparse.ArgumentParser()
parser.add_argument('youtube_id', help='YouTube video ID')
args = parser.parse_args()

def download_video(yt_id):
	#if not os.path.exists('%s/%s.mp4' % (scv.config['path']['videos'], yt_id)):
	scv.log('Downloading video: %s' % yt_id, scv.DEBUG_NOTICE)
	yt_opts = dict(
		playlistend = 50,
		max_downloads = 5,
		matchtitle = '[Vv][Ss].+([Ss]m[a4]sh\s*4|[Ss][Ss][Bb]4)',
		format = '134/135/mp4[acodec=none]',
		#quiet = True,
		#forcetitle = True,
		outtmpl = '%s/%s.mp4' % (scv.config['path']['videos'], '%(id)s'))
	with youtube_dl.YoutubeDL(yt_opts) as ytdl:
		return ytdl.download([yt_id])
	#else:
	#	scv.log('Video already exists, skipping: %s' % yt_id, scv.DEBUG_NOTICE)

if __name__ == '__main__':
	download_video(args.youtube_id)
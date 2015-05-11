from __future__ import unicode_literals

import os.path
import json
import io
import argparse
import youtube_dl
import SmashCVCore

# CLI
parser = argparse.ArgumentParser(description='Extract meaningful information from Super Smash Bros. for Wii U videos.')
parser.add_argument('-ytid', 
	#dest='ytid',
	help='YouTube video ID',
	#default='jcsKByup_Tw'
	default='nCte6rhdAqs'
	)
parser.add_argument('-o', dest='outfile',
	help='Output destination')
args = parser.parse_args()

# Youtube options
yt_local_path = 'videos/%s.mp4'
yt_options = {
	'format': '132/133/135/mp4[acodec=none][height=240]/mp4[acodec=none][height<=360]',
	'outtmpl': yt_local_path % '%(id)s'
}
yt_remote_uri = 'https://www.youtube.com/watch?v=%s'

# Download video
if(not os.path.exists(yt_local_path % args.ytid)):
	print 'downloading video'
	with youtube_dl.YoutubeDL(yt_options) as ytdl:
		ytdl.download([yt_remote_uri % args.ytid])

events = SmashCVCore.process_video(yt_local_path % args.ytid)
with io.open('output/%s.json' % args.ytid, 'w', encoding='utf-8') as f:
	f.write(unicode(json.dumps(events, ensure_ascii=False)))
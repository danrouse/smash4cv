from __future__ import unicode_literals

import os.path
import youtube_dl
import SmashCVCore

# Youtube options
#ytVideoID = '3tai70AIoe4'
ytVideoID = 'jcsKByup_Tw'
ytVideoPath = 'videos/%s.mp4'
ytOptions = {
	'format': '132/133/135/mp4[acodec=none][height=240]/mp4[acodec=none][height<=360]',
	'outtmpl': ytVideoPath % '%(id)s'
}
ytWatchURI = 'https://www.youtube.com/watch?v=%s'

# Download video
if(not os.path.exists(ytVideoPath % ytVideoID)):
	print 'downloading video'
	with youtube_dl.YoutubeDL(ytOptions) as ytdl:
		ytdl.download([ytWatchURI % ytVideoID])

SmashCVCore.processVideo(ytVideoPath % ytVideoID)
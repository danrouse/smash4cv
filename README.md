# Smash 4 Computer Vision

A tool to extract meaningful information from videos of Super Smash Bros. for Wii U: players, fighters, and game events. smash4cv automatically extracts game stats (JSON) and highlight reels from each video.


## Dependencies
- Python 3.4
- OpenCV 3.0-rc1 (with contrib)
- youtube_dl
- numpy

## Usage
- Download videos
	
	- `scv-download.py --id=8uqAAppaCa4`
	- `scv-download.py --user=VideoGameBootCamp --filter=[Vv][Ss].+[sS]mash\s*4`

- Process downloaded videos

	`scv-process.py`
	`scv-process.py --video=videos/8uqAAppaCa4.mp4`
	`scv-process.py --video=videos/8uqAAppaCa4.mp4 --retrain=4460`

## Configuration
Values in [config.json](config.json) can be modified to achieve better detection results.

## Development
This software has been built on Ubuntu 15.04, but should run on other systems - granted you can get OpenCV 3.0 compiled for Python 3.

### Training

This program uses two methods of region detection:

	- *matchTemplate* (`cv2.matchTemplate`) detects game states, using the templates in config.path.templates (default: ./templates). These templates should be a fairly low resolution, as the source frames are downsampled to find the best match.

	- *KNearest* (`kNN.kNN` < `cv2.ml.KNearest`) detects in-game state: names, digits, and stages.

### Roadmap and Known Issues
- Doesn't work with all Smash streamers. Need to be able to detect from any video layout with minimal tweaking.

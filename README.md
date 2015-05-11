# Smash 4 Computer Vision

A tool to extract meaningful information from videos of Super Smash Bros. for Wii U. Why watch people play video games when a computer can do the same?

## Dependencies
- OpenCV/cv2
- numpy
- youtube_dl

## Usage
`SmashCV.py [-o OUTFILE] ytid`
`ytid` is the YouTube ID of the input video
`OUTFILE` is the path to generated JSON

## Known Issues
- Training data was generated in Photoshop, not from live game data. Unit testing the OCR is complicated without a proper training set.
- Only tested at the moment with VGBC replays. Have to add in dynamic scaling/template matching to support other streams (inner video may be different size) but should be simple! **Famous last words.**
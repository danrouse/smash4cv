# Smash 4 Computer Vision

A tool to extract meaningful information from videos of Super Smash Bros. for Wii U. Why watch people play video games when a computer can do the same?

## Dependencies
- OpenCV/cv2
- numpy
- youtube_dl

## Usage
`SmashCV.py [-o OUTFILE] ytid`

Where:
- `ytid` is the YouTube ID of the input video
- `OUTFILE` is the path to generated JSON

## Known Issues
- Training data was generated in Photoshop, not from live game data. Unit testing the OCR is complicated without a proper training set.
- Only tested at the moment with VGBC replays. Have to add in dynamic scaling/template matching to support other streams (inner video may be different size) but should be simple! *Famous last words.*

## Training
### kNN
k-NearestNeighbor is used to quickly recognize digits in the in-game state of the software. A random seed was used to generate test data which was imported into Photoshop and saved as a grayscale PNG, with font styles and noise, using the in-game numbers font (DF Gothic SU)

### Tesseract
Tesseract is used for general text OCR. Training data was created following [https://code.google.com/p/tesseract-ocr/wiki/TrainingTesseract3](this guide), using the data in ./training/
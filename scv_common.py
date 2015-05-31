from enum import Enum
import json
import cv2

config = json.load(open('./config.json'))

class State(Enum):
	unknown	= 1
	loading	= 2
	ingame	= 3
	quit	= 4

DETECT_DEAD 	= -1
DETECT_UNKNOWN	= -2

# Python's Enums are not great for bitmasking operations
DEBUG_NONE		= 0b00000
DEBUG_NOTICE	= 0b00001
DEBUG_VIDEO		= 0b00010
DEBUG_EVENTS	= 0b00100
DEBUG_PERF		= 0b01000
DEBUG_DETECT	= 0b10000
DEBUG_ALL		= 0b11111

# Logging and debugging
debug_log = []
debug_level = DEBUG_NONE

def log(data, log_level = DEBUG_NOTICE):
	debug_log.append((log_level, data))
	if debug_level & log_level:
		if type(data) is str:
			print(data)
		else:
			print(': '.join([str(d) for d in data]))

def benchmark(fn):
	def timer(*args, **kw):
		if not debug_level & DEBUG_PERF:
			return fn(*args, **kw)

		t1 = cv2.getTickCount()
		ret = fn(*args, **kw)

		t2 = cv2.getTickCount()
		dt = (t2 - t1) / cv2.getTickFrequency()
		log((fn.__name__, '%.3fms' % (dt * 1000)), DEBUG_PERF)

		return ret

	return timer
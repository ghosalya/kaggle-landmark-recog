"""
PURPOSE: provide helper functions for image extraction program
"""

import requests
import time
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import math
import operator
from functools import reduce

def getImage(id, link, dir):
	"""
	PURPOSE: requests image from link and saves at specified directory
	"""
	try:
		r = requests.get(link, stream=True)
		if r.status_code != 200:
			print(">>> Request FAILURE")
			print("ERROR: Missing image at %s" % link)
			with open("./logs/missing_links.csv", "a") as output:
				output.write("%s, %s, %s\n" %(time.time(),id,link))
			return 0
		else:
			print(">>> Request SUCCESS")
			with open(dir+"%s.jpg" %id,"wb") as output:
				r.raw.decode_content = True
				shutil.copyfileobj(r.raw, output)
				try:
					size = output.tell()
					print(">>> Write SUCCESS")
					return int(size)
				except Exception:
					print(">>> Write FAILURE")
					with open("./logs/missing_links.csv", "a") as output:
						output.write("%s, %s, %s\n" %(time.time(),id,link))
	except Exception:
		print(">>> Request FAILURE")
		print("ERROR: Missing image at %s" % link)
		with open("./logs/missing_links.csv", "a") as output:
			output.write("%s, %s, %s\n" %(time.time(),id,link))
		return 0
	

			

# getImage("blank","http://static.panoramio.com/photos/original/70761397.jpg","/run/media/dekatria/My Passport/")

def getProgress(i, n, size=None):
	"""
	PURPOSE: takes in current index in dataset and outputs script progress string.
	"""
	total = float((i+1)/n)
	percentage = round(100*total, 2)
	if size:
		mbSize = round(size/1048576, 2)
		return "[ %s %% // %s MB]" %(percentage, mbSize)
	else:
		return "[ %s %% ]" %percentage

# print(getProgress(52,823,getImage("blank","http://static.panoramio.com/photos/original/70761397.jpg","/run/media/dekatria/My Passport/")))

def compareImages(local, remote, threshold=0.1):
	"""
	PURPOSE: compares image from link with image stored locally and outputs boolean for similarity
	"""
	loc = Image.open(local).histogram()
	try:
		r = requests.get(remote, stream=True)
		if r.status_code != 200:
			return "unverified"
		else:
			r.raw.decode_content = True
			rem = Image.open(r.raw).histogram()
			rms = math.sqrt(reduce(operator.add,
				map(lambda a,b: (a-b)**2, loc, rem))/len(loc))
			if rms <= threshold:
				return None
			else:
				return "incorrect"
	except Exception as ins:
		return "unverified"

# print(compareImages("/home/dekatria/Downloads/70761397.jpg","http://static.panoramio.com/photos/original/70761397.jpg"))

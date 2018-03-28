"""
PURPOSE: provide helper functions for image extraction program
"""

import requests
import time
import matplotlib.pyplot as plt
import shutil

def getImage(id, link, dir):
	"""
	requests image from link and saves at specified directory
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
		pass
	except Exception:
		print(">>> Request FAILURE")
		print("ERROR: Missing image at %s" % link)
		with open("./logs/missing_links.csv", "a") as output:
			output.write("%s, %s, %s\n" %(time.time(),id,link))
		return 0
	

			

# getImage("blank","http://static.panoramio.com/photos/original/70761397.jpg","/run/media/dekatria/My Passport/")

def getProgress(i, n, size):
	total = float((i+1)/n)
	percentage = round(100*total, 2)
	mbSize = round(size/1048576, 2)
	return "[ %s %% // %s MB]" %(percentage, mbSize)

# print(getProgress(52,823,getImage("blank","http://static.panoramio.com/photos/original/70761397.jpg","/run/media/dekatria/My Passport/")))


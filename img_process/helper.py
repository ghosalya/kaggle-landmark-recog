"""
PURPOSE: provide helper functions for image processing tasks
"""

from random import shuffle
from math import floor
from resizeimage import resizeimage
from PIL import Image
import os 

def resize_img(input_filename, output_filename, 
			   size=128, overwrite=False):
	'''
	PURPOSE: resize image to a square image of (size)x(size)
	'''
	if os.path.exists(output_filename) and not overwrite:
		# if resized image exists, dont do anything
		return False
	try:
		with open(input_filename, 'r+b') as f:
			with Image.open(f) as image:
					cover = resizeimage.resize_cover(image, (size, size))
					cover.save(output_filename, image.format)
	except Exception as e:
		print(str(e))

def binImages(imgList, n):
	"""
	PURPOSE: generate random sets for testing and validation
	"""
	shuffle(imgList)
	return imgList[:-floor(len(imgList)/n)], imgList[-floor(len(imgList)/n):]

# print(binImages(["%s.jpg" %i for i in range(10)], 5))
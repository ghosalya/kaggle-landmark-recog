"""
PURPOSE: extract image from the links given in the dataset.
"""

import helper
import pandas as pd
import sys

def main(file, dir, start=0):
	dataset = pd.read_csv("./data/%s" % file, dtype="str")
	print("########## START SCRIPT ##########")
	n = dataset.shape[0]
	size = 0
	for i in range(start,n):
		imgId = dataset["landmark_id"][i] + "-" + dataset["id"][i]
		size += helper.getImage(imgId, dataset["url"][i], dir)
		print("%s %s : %s" %(helper.getProgress(i, n, size), i, imgId))
	print("########## END SCRIPT ##########")

if __name__ == "__main__":
	if len(sys.argv) == 2:
		main(sys.argv[0], sys.argv[1])
	elif len(sys.argv) == 3:
		main(sys.argv[0], sys.argv[1], sys.argv[2])
	else:
		raise IndexError("Invalid number of parameters")
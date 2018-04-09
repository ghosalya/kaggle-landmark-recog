"""
PURPOSE: checks if all images are downloaded correctly and logs missing images
"""

import helper
import pandas as pd
import sys
import os
import time

def main(file, dir, marker="ind", content=False):
	dataset = pd.read_csv("./data/%s" % file, dtype="str")
	fileList = os.listdir(dir)
	print("########## START SCRIPT ##########")
	print(fileList)
	count = 0
	n = dataset.shape[0]
	if marker == "ind":
		for i in range(n):
			imgId = "%s.jpg" % i
			if imgId in fileList:
				print("%s %s : %s FOUND" %(helper.getProgress(i, n), i, imgId))
				if content:
					result=helper.compareImages(dir + imgId, dataset["url"][i])
					if not result:
						print("%s CORRECT" % imgId)
					else:
						print("%s MISMATCH: %s" %(imgId, result))
						with open("./logs/missing_links.csv", "a") as output:
							output.write("%s, %s, %s, %s\n" %(time.time(),
								dataset["landmark_id"][i] + "-" + dataset["id"][i],
								dataset["url"][i]),
								result)	

			else:
				print("%s %s : %s NOT FOUND" %(helper.getProgress(i, n), i, imgId))
				count += 1
				with open("./logs/missing_links.csv", "a") as output:
					output.write("%s, %s, %s, missing\n" %(time.time(), 
						dataset["landmark_id"][i] + "-" + dataset["id"][i],
						dataset["url"][i]))	
	elif marker == "idid":
		for i in range(n):
			imgId = "%s-%s.jpg" %(dataset["landmark_id"][i], dataset["id"][i])
			if imgId in fileList:
				print("%s %s : %s FOUND" %(helper.getProgress(i, n), i, imgId))
				if content:
					result=helper.compareImages(dir + imgId, dataset["url"][i])
					if not result:
						print("%s CORRECT" % imgId)
					else:
						print("%s MISMATCH: %s" %(imgId, result))
						with open("./logs/missing_links.csv", "a") as output:
							output.write("%s, %s, %s, %s\n" %(time.time(),
								dataset["landmark_id"][i] + "-" + dataset["id"][i],
								dataset["url"][i]),
								result)	
			else:
				print("%s %s : %s NOT FOUND" %(helper.getProgress(i, n), i, imgId))
				count += 1
				with open("./logs/missing_links.csv", "a") as output:
					output.write("%s, %s, %s, missing\n" %(time.time(), 
						dataset["landmark_id"][i] + "-" + dataset["id"][i],
						dataset["url"][i]))	

	print("Number of Missing Images: %s" % count)
	print("########## END SCRIPT ##########")

if __name__ == "__main__":
	if len(sys.argv) == 3:
		main(sys.argv[1], sys.argv[2])
	elif len(sys.argv) == 4:
		main(sys.argv[1], sys.argv[2], sys.argv[3])
	elif len(sys.argv) == 5:
		main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	else:
		for i in range(len(sys.argv)):
			print(sys.argv[i])
		raise IndexError("Invalid number of parameters")

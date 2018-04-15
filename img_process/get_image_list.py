"""
PURPOSE: from image directory, get list of images with information
"""

import os
import pandas as pd
import sys

def main(file, target, mainDir, secDir=None):
	print("########## START SCRIPT ##########")
	data = pd.read_csv(file)
	train = os.listdir(mainDir)
	if secDir:
		valid = os.listdir(secDir)
	else:
		valid = list(set(data.file_name.tolist())-set(train))
	print(">>> Import SUCCESS")
	print(">>> Training log BEGIN")
	trainFile = data[data.file_name.isin(train)]
	trainFile.to_csv("%straining.csv" % target, columns=["id", "landmark_id", "file_name"])
	print(">>> Training log END")

	print(">>> Validation log BEGIN")
	validFile = data[data.file_name.isin(valid)]
	validFile.to_csv("%svalidation.csv" % target, columns=["id", "landmark_id", "file_name"])
	print(">>> Validation log END")
	print("########## END SCRIPT ##########")

if __name__ == '__main__':
	if len(sys.argv) == 4:
		main(sys.argv[1], sys.argv[2], sys.argv[3])
	elif len(sys.argv) == 5:
		main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	else:
		for i in range(len(sys.argv)):
			print(sys.argv[i])
		raise IndexError("Invalid number of parameters")

# python img_process/get_image_list.py "/home/dekatria/Documents/projects/201802.university.t7.computer_vision.project/kaggle-landmark-recog/logs/train80.csv" "/home/dekatria/Documents/projects/201802.university.t7.computer_vision.project/kaggle-landmark-recog/data/resized/500/set-0/" "/home/dekatria/Documents/projects/201802.university.t7.computer_vision.project/kaggle-landmark-recog/data/resized/500/set-0/train/"
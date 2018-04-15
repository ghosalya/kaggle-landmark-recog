'''
PURPOSE: Resize the original dataset to set of images
with specific size, and produce corresponding .csv
'''
import pandas as pd 
from PIL import Image
import helper
import os
import sys

def main(dir, file_name, size=128, folds=10, sets=1):
	print("########## START SCRIPT ##########")
	data = pd.read_csv(file_name)
	outputName = "./data/resized/%s/" % size
	fileList = data["file_name"].tolist()
	print(">>> Import SUCCESS")
	for i in range(int(sets)):
		print(">>> Set %s BEGIN" % i)
		if not os.path.exists("%sset-%s/train/" %(outputName, i)):
			os.makedirs("%sset-%s/train/" %(outputName, i))
		if not os.path.exists("%sset-%s/validate/" %(outputName, i)):
			os.makedirs("%sset-%s/validate/" %(outputName, i))
		
		train, valid = helper.binImages(fileList, int(folds))

		print(">>> Training log BEGIN")
		trainFile = data[data.file_name.isin(train)]
		trainFile.to_csv("%sset-%s/training.csv" %(outputName, i), columns=["id", "landmark_id", "file_name"])
		print(">>> Training log END")

		print(">>> Validation log BEGIN")
		validFile = data[data.file_name.isin(valid)]
		validFile.to_csv("%sset-%s/validation.csv" %(outputName, i), columns=["id", "landmark_id", "file_name"])
		print(">>> Validation log END")

		print(">>> Training resize BEGIN")
		for file in train:
			source_file = dir + file 
			dest_file = "%sset-%s/train/%s" %(outputName, i, file)
			success = helper.resize_img(source_file, dest_file, size = int(size))
			print("[SET %s // Training // %s %% ] %s processed" %(i, round((train.index(file)+1)/len(train)*100,2), file))
		
		print(">>> Training resize END")

		print(">>> Validation set resize BEGIN")
		for file in valid:
			source_file = dir + file 
			dest_file = "%sset-%s/valid/%s" %(outputName, i, file)
			success = helper.resize_img(source_file, dest_file, size = int(size))
			print("[SET %s // Validation // %s %% ] %s processed" %(i, round((valid.index(file)+1)/len(valid)*100,2), file))
		
		print(">>> Validation resize END")

		print(">>> Set %s END" % i)
	print("########## END SCRIPT ##########")

if __name__ == '__main__':
	if len(sys.argv) == 4:
		main(sys.argv[1], sys.argv[2], size=sys.argv[3])
	elif len(sys.argv) == 5:
		main(sys.argv[1], sys.argv[2], size=sys.argv[3], folds=sys.argv[4])
	elif len(sys.argv) == 6:
		main(sys.argv[1], sys.argv[2], size=sys.argv[3], folds=sys.argv[4], sets=sys.argv[5])
	else:
		for i in range(len(sys.argv)):
			print(sys.argv[i])
		raise IndexError("Invalid number of parameters")

# python img_process/resize_images.py "/run/media/dekatria/My Passport/kaggle_dataset/google_recognition/train/" "/home/dekatria/Documents/projects/201802.university.t7.computer_vision.project/kaggle-landmark-recog/logs/train80.csv" 500
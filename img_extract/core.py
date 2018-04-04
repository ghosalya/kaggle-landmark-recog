import helper
import pandas as pd

def main(start, file, dir):
	dataset = pd.read_csv("./data/%s" % file, dtype="str")
	print("########## START SCRIPT ##########")
	# table = {}
	n = dataset.shape[0]
	size = 0
	for i in range(start,n):
		# ldmkId = dataset["landmark_id"][i]
		
		imgId = dataset["landmark_id"][i] + "-" + dataset["id"][i]

		# if ldmkId in table.keys():
		# 	table[ldmkId] += 1
		# 	imgId = ldmkId + "-6-" + str(table[ldmkId])
		# else:
		# 	table[ldmkId] = 1
		# 	imgId = ldmkId + "-6-1"

		size += helper.getImage(imgId, dataset["url"][i], dir)
		print("%s %s : %s" %(helper.getProgress(i, n, size), i, imgId))
	print(">>> Log BEGIN")
	# with open("./logs/total_count.csv", "a") as output:	
	# 	for key in table:
	# 		output.write("%s, %s" %(key, table[key]))
	print(">>> Log END")
	print("########## END SCRIPT ##########")

# main("recognition/train.csv", "/run/media/dekatria/My Passport/kaggle_dataset/google_recognition/train/")
main(338324,"recognition/train.csv", "/home/dekatria/Downloads/kaggle_dataset/google_recognition/train/")
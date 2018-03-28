import helper
import pandas as pd

def main(file, dir):
	dataset = pd.read_csv("./data/%s" % file, dtype="str")
	print("########## START SCRIPT ##########")
	table = {}
	n = dataset.shape[0]
	size = 0
	for i in range(20166,n):
		ldmkId = dataset["landmark_id"][i]
		if ldmkId in table.keys():
			table[ldmkId] += 1
			imgId = ldmkId +"-2-"+ str(table[ldmkId])
		else:
			table[ldmkId] = 1
			imgId = ldmkId + "-2-1"

		size += helper.getImage(imgId, dataset["url"][i], dir)
		print(helper.getProgress(i, n, size)+" "+imgId)
	print(">>> Log BEGIN")
	with open("./logs/total_count.csv", "a") as output:	
		for key in table:
			output.write("%s, %s" %(key, table[key]))
	print(">>> Log END")
	print("########## END SCRIPT ##########")

main("recognition/train.csv", "/run/media/dekatria/My Passport/kaggle_dataset/google_recognition/train/")
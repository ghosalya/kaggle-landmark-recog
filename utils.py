'''
Utility Functions
'''
import cv2, os
import pandas as pd 
import numpy as np

def get_dataset(dataset="196", batch=5000):
	'''
	Function to load datasets 

	data/
		train.csv 			---	original train.csv
		test.csv 			--- original test.csv
		train_images/ 		--- original set of images for train.csv
		test_images/ 		--- original set of images for test.csv
		{size}/ 			--- folder correponds to image size
			train.csv 			e.g. 196/ is directory for images
			test.csv 			size of 196x196
			train/
			test/

	'''
	data_dir="./data/{}/train/".format(dataset)
	data_csv = "./data/{}/train.csv".format(dataset)
	df = pd.read_csv(data_csv).sample(batch)
	data = []
	nudf = df.copy()
	i = 0
	for idx, row in df.iterrows():
		i+=1
		# while True:
		img_path = data_dir + row['file_name']
		im = cv2.imread(img_path)
		if im is None:
			print(row['file_name'],'exists:', os.path.exists(img_path))
			print('dropping')
			nudf = nudf.drop(idx)
		else:
			data.append(im.tolist())
			if(i % 100 == 0):
				print("Loading ",i)
	x = np.asarray(data)
	y = nudf['landmark_id'].values
	return x, y

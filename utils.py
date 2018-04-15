'''
Utility Functions
'''
import cv2, os
import pandas as pd 
import numpy as np

DATASET_DEFAULT = "128"

def get_dataset(dataset=DATASET_DEFAULT, batch=5000, index=None):
	'''
	Function to load datasets 

	if index is given a list, returns loaded image of selected list

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
	df = get_data_csv(dataset)

	# does either slicing or sampling
	if index is None:
		df = df.sample(batch)
	elif isinstance(index, (list, tuple)):
		df = df.iloc[index]
	else:
		df = df[index:index+batch]

	return get_dataset_images(df, dataset)


def get_data_csv(dataset):
	'''
	Low-level implementation
	to give more control to trainer
	'''
	data_csv = "./data/{}/train160.csv".format(dataset)
	df = pd.read_csv(data_csv)
	return df	


def get_dataset_length(dataset=DATASET_DEFAULT):
	'''
	simply returns length of dataset csv
	'''
	data_csv = "./data/{}/train160.csv".format(dataset)
	df = pd.read_csv(data_csv)
	return len(df)


def get_dataset_images(df, dataset=DATASET_DEFAULT):
	'''
	Function to load images
	given the pandas dataframe
	(low-level implementation)
	'''
	data_dir="./data/{}/train/".format(dataset)
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
			# if(i % 100 == 0):
			# 	print("Loading ",i)
	x = np.asarray(data)
	y = nudf['landmark_id'].values
	return x, y


def print_tele(string):
	'''
	uses telegram_send
	if telegram_send is not setup,
	should act like a normal print

	note that unlike normal print,
	this function only takes 1 string
	at a time.
	'''
	print(string)
	try:
		import telegram_send
		telegram_send.send(messages=[string])
	except:
		print("Failed to send Telegram")
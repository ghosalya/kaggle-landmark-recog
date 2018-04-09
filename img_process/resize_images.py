'''
Resize the original dataset to set of images
with specific size, and produce corresponding .csv
'''
import pandas as pd 
from PIL import Image
from resizeimage import resizeimage
import os

def resize_img(input_filename, output_filename, 
			   size=128, overwrite=False):
	'''
	Resize image to a square image of (size)x(size)
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

def main(size):
	data=pd.read_csv('./data/train.csv')
	i = 0
	img_no = len(data)
	new_dataset_list = []
	if not os.path.exists('./data/{}/train/'.format(size)):
		# os.makedir('./data/{}/'.format(size))
		os.makedirs('./data/{}/train/'.format(size))
	for idx, dat in data.iterrows():
		source_file = './data/train_images/{}.jpg'.format(i)
		dest_file = './data/{}/train/{}.jpg'.format(size, i)
		success = resize_img(source_file, dest_file)
		if success:
			newrow = [dat["id"],  dat["landmark_id"], str(i)+".jpg"]
			new_dataset_list.append(newrow)

		if i % 100 == 0:
			print("{} / {} resized".format(i, img_no))
		i += 1
	newdata = pd.DataFrame(thelist, columns=("id","landmark_id","file_name"))
	newdata.to_csv("./data/{}/train.csv".format(size))

if __name__ == '__main__':
	main(128)
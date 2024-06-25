import torch
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import skimage
import os
import random
import matplotlib.pyplot as plt



def patch_info(filenames, img_size, patch_size, stride, augmentation = ["flip_vertical", "flip_horizontal", "rotation90", "rotation180"]):

	""" Create a data frame with the patch infos : original image name, patch coordinates, data augmentation...

	Keyword arguments:
	filenames -- list of the dataset image names (list of str)
	img_size --  size of the original images (tuple of int) ex: (1024, 1024)
	patch_size -- size of the patch (tuple of int) ex: (64,64)
	stride -- stride to apply to overlapping patches (int)
	augmentations -- list of augmentations to apply to the images ("flip_vertical", "flip_horizontal", "rotation90", "rotation180")
	"""

	data_dict = {"filename": [], "augmentation": [], "patch_xmin": [], "patch_xmax": [], "patch_ymin": [], "patch_ymax": []}
	
	for file in filenames:
	
		# Write augmentations
		if augmentation:
			
			coords = divide_into_patches(img_size, patch_size, stride)

			if "none" not in augmentation:
				augmentation.append("none")

			for a in augmentation:		
				for c in coords:
					data_dict["filename"].append(file)
					data_dict["augmentation"].append(a)
					data_dict["patch_xmin"].append(c[0])
					data_dict["patch_xmax"].append(c[1])
					data_dict["patch_ymin"].append(c[2])
					data_dict["patch_ymax"].append(c[3])

		else:
			
			# Get patch coords
			coords = divide_into_patches(img_size, patch_size, stride)
			for c in coords:
				data_dict["filename"].append(file)
				data_dict["augmentation"].append(a)
				data_dict["patch_xmin"].append(c[0])
				data_dict["patch_xmax"].append(c[1])
				data_dict["patch_ymin"].append(c[2])
				data_dict["patch_ymax"].append(c[3])
					
	data_frame = pd.DataFrame.from_dict(data_dict)
	return data_frame



def divide_into_patches(img_size, patch_size, stride):

	""" Return the coordinates patches in the image.

	Keyword arguments : 
	img_size --  size of the original images (tuple of int) ex: (1024, 1024)
	patch_size -- size of the patch (tuple of int) ex: (64,64)
	stride -- stride to apply to overlapping patches (int) 
	"""

	if img_size[0] == patch_size[0] and img_size[1] == patch_size[1]:
		patch_list = [[0, img_size[0], 0, img_size[1]]]
	else:

		patch_list = []

		sx = img_size[0]
		sy = img_size[1]
		
		if stride < patch_size[0]:

			nbpx = ((sx - patch_size[0]) // stride) + 1
			nbpy = ((sy - patch_size[1]) // stride) + 1

			for i in range(nbpx):
				for j in range(nbpy):
					patch_list.append([stride * i, stride * i + patch_size[0], stride * j,  stride * j + patch_size[1]])

			# Add the border patches to complete the image
			if (sx - patch_size[0]) % stride != 0:
				for i in range(nbpy):
					patch_list.append([sx-patch_size[0], sx, stride * i , stride * i + patch_size[1]])
				patch_list.append([sx-patch_size[0], sx, sy - patch_size[1], sy])

			if (sy - patch_size[1]) % stride != 0:
				for i in range(nbpx):
					patch_list.append([stride * i, stride * i + patch_size[0], sy - patch_size[1], sy])

		else:

			nbpx = sx // patch_size[0]
			nbpy = sy // patch_size[1]

			if sx % patch_size[0] == 0:
				nbpx+=1
			if sy % patch_size[1] == 0:
				nbpy+=1

			for i in range(nbpx-1):
				for j in range(nbpy-1):
					patch_list.append([i*patch_size[0], (i+1)*patch_size[0], j*patch_size[1], (j+1)*patch_size[1]])

			# Add the border patches to complete the image
			if sx % patch_size[0] != 0:
				for i in range(nbpy-1):
					patch_list.append([sx-patch_size[0], sx, i*patch_size[1], (i+1)*patch_size[1]])

			if sy % patch_size[1] != 0:
				for i in range(nbpx-1):
					patch_list.append([i*patch_size[0], (i+1)*patch_size[0], sy-patch_size[1], sy])

	return np.array(patch_list)




class Dataset(Dataset):

	# Dataset class
	def __init__(self, data_frame, images, nametags, patch_size):

		"""
		Keyword arguments:
		data_frame -- pandas data frame of patch info (cf patch_info function)
		images -- list of dictionary containind the images of the different datasets
		nametags -- list of names to use for the output tensors
		patch_size -- the output patch size for input in the nn (tuple of int)
		"""

		self.data_frame = data_frame
		self.images = images
		self.nametags = nametags
		self.patch_size = patch_size
		

	def __len__(self):
		return self.data_frame.shape[0]

	def __getitem__(self, idx):

		tensor_dict = {}
		for i in range(len(self.nametags)):

			filename = self.data_frame["filename"].iloc[idx]


			aug = self.data_frame["augmentation"].iloc[idx]
			xmin, xmax, ymin, ymax = self.data_frame["patch_xmin"].iloc[idx], self.data_frame["patch_xmax"].iloc[idx], self.data_frame["patch_ymin"].iloc[idx], self.data_frame["patch_ymax"].iloc[idx]
			
			# Crop patch 
			patch = self.images[i][filename][:, xmin:xmax, ymin:ymax]
		
			# Resize
			tr = transforms.Resize(self.patch_size, max_size=None, antialias=True)
			patch = tr(patch)
			

			# Apply augmentation
			if aug != "none":
				if aug == "flip_vertical":
					tr = transforms.RandomVerticalFlip(p=1)
				elif aug == "flip_horizontal":
					tr = transforms.RandomHorizontalFlip(p=1)
				elif aug == "rotation90":
					tr = transforms.RandomRotation((90,90))
				elif aug =="rotation180":
					tr = transforms.RandomRotation((180,180))
				
				patch = tr(patch)

			tensor_dict[self.nametags[i]] = patch


		return tensor_dict



def split_dataset(folder, names, percentage, output_folder = "", cond = None, write = True):

	file_list = []
	for root, dirs, files in os.walk(folder):
		for file in files:
			if cond is not None:
				if cond in file:
					file_list.append(file)
			else:
				file_list.append(file)

	# Randomize file list
	random.shuffle(file_list)
	filenames = {}

	if len(percentage) == 1:
		filenames[name] = file_list
	else:
		nb = len(file_list)

		idx = []
		for i in range(len(percentage) - 1):
			if len(idx) == 0:
				idx.append(int(nb*percentage[i]/100))
			else:
				idx.append(idx[-1] + int(nb*percentage[i]/100))
		idx.append(nb)
		
		filenames[names[0]] = file_list[:idx[0]]
		for i in range(0, len(idx) -1):
			filenames[names[i+1]] = file_list[idx[i]:idx[i+1]]

	print("Spliting dataset : ", [len(f) for f in filenames.values()])

	if write:

		os.makedirs(output_folder, exist_ok=True)
		
		for n in filenames.keys():
			file = open(output_folder + "filenames_" + n + ".txt", "w")

			for f in filenames[n]:
				 file.write(f + "\n")

			file.close() 

	return filenames


def load_file_list(file_path):

	filenames = []
	f = open(file_path, 'r')
	lines = f.readlines()

	for line in lines:
		filenames.append(line.replace("\n", ""))

	return filenames




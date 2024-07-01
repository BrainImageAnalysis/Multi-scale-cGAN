import torch 
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import skimage
import torchvision.utils as vutils
from torch.utils.data import Subset
from PIL import Image, ImageOps
from math import pi, cos
import gc
import pickle
import random
import torchvision.transforms as transforms

from torchmetrics.image.fid import FrechetInceptionDistance

from models import *
from data import *


class MultiResGenerator:

	def __init__(self, output_folder, nb_res, real_patch_size, model_patch_size, model_type, itrs, augmentation, stride, lr, lbd, nz, batch_size, n_gt_channels = None, n_img_channels = None):

		self.output_folder = output_folder

		# Create output folder if not existing 
		self.create_dir(output_folder)
		for i in range(nb_res):
			self.create_dir(output_folder + "/scale_" + str(i) + "/")

		self.real_patch_size = real_patch_size
		self.model_patch_size = model_patch_size
		self.model_type = model_type
		self.nb_res = nb_res
		self.itrs = itrs
		self.augmentation = augmentation
		self.stride = stride
		
		self.lr = lr
		self.nz = nz
		self.lbd = lbd
		self.batch_size = batch_size

		self.models = [None] * nb_res

		self.images = {}
		self.datasets = {}

		self.z = {}

		# Write parameters 
		self.write_parameters()
		self.n_gt_channels = n_gt_channels
		self.n_img_channels = n_img_channels



	def load_images(self, img_folder, data_type, dataset_name, equalize, resize = None, data_range = [-1,1], file_list = [], repeat = 1, write = True):

		""" Load the images for scale 0

		Keytword arguments:
		img_folder -- list of dataset folder paths (str)
		data_type -- list of type of data ("gt" or "img")
		dataset_name -- name of the image dataset (str) ex: "train"
		data_range -- target image intensity range (list of int) ex: [0,1]
		equalize -- list of whether to equalize the images or not (bool)
		file_list -- list of files to include in the imageset
		repeat -- number of repetition of the same image in the dataset
		write -- wether to write some of the images in the dataset (bool)
		"""

		if dataset_name not in list(self.images.keys()):
			self.images[dataset_name] = {}

		for i in range(len(img_folder)):
			if data_type[i] not in list(self.images[dataset_name].keys()):
				self.images[dataset_name][data_type[i]] = {}

			if len(file_list) == 0:
				# Get all files 
				file_list = []
				for root, dirs, files in os.walk(img_folder[i]):
					for file in files:
						file_list.append(file)
			
			print("Loading " + str(len(file_list)*repeat) + " images from folder " + img_folder[i])

			for file in file_list:
				
				img = Image.open(img_folder[i] + file)
				if equalize[i]:
					img = ImageOps.equalize(img)
				img = np.array(img).astype(float)

				# Normalize data
				img = (data_range[1] - data_range[0])*((img - img.min())/(img.max() - img.min())) + data_range[0]
				img[np.isnan(img)] = data_range[0]

				if len(img.shape) > 2: # RGB image
					img = np.vstack([np.expand_dims(img[:,:,i], axis=0) for i in range(3)])
				else: # gray scale image
					img = np.expand_dims(img, axis=0)

				# Change to tensor
				img = img.astype(np.float32)
				img = torch.from_numpy(img)

				# Resize
				if resize is not None:
					resize_tr = transforms.Resize(resize, max_size=None, antialias=True)
					img = resize_tr(img)

				for r in range(repeat):
					if repeat > 1:
						n = file.replace(".png", str(r) + ".png")
					else:
						n = file
					if n in list(self.images[dataset_name][data_type[i]].keys()):
						self.images[dataset_name][data_type[i]][n] = torch.cat((self.images[dataset_name][data_type[i]][n], img), dim = 0)
					else:
						self.images[dataset_name][data_type[i]][n] = img
		if self.n_gt_channels is None:
			self.n_gt_channels = self.images[dataset_name]["gt"][list(self.images[dataset_name]["gt"].keys())[0]].shape[0]
		if self.n_img_channels is None:
			self.n_img_channels = self.images[dataset_name]["img"][list(self.images[dataset_name]["img"].keys())[0]].shape[0]

		# Write list of files
		#self.write_dataset_file(dataset_name)

		# Write images
		if write:
			for i in range(len(data_type)):
				self.write_images(0, dataset_name, data_type[i], 5)


	def merge_imagesets(self, dataset_name1, dataset_name2):

		# Rename dict keys to avoid doubles
		filenames = list(self.images[dataset_name2].keys())
		for typ in filenames:
			for k in self.images[dataset_name2][typ]:
				self.images[dataset_name2][typ][k.replace(".png", "2.png")] = self.images[dataset_name2][typ].pop(k)

		# Merge dicts
		self.images[dataset_name1].update(self.images[dataset_name2])



	def create_dataset(self, res_id, dataset_name, train = True, write = True):

		""" Creates a dataset from loaded images

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the image dataset (str) ex: "train"
		train -- True for a training dataset, False for a test dataset
		"""
		print("Creating dataset")

		if dataset_name not in list(self.datasets.keys()):
			self.datasets[dataset_name] = {}

		# Dataset content
		add_noise = []
		if res_id == 0:
			if train:
				input_data = ["gt", "img"]
				images = [self.images[dataset_name]["gt"], self.images[dataset_name]["img"]]

			else:
				input_data = ["gt"]
				images = [self.images[dataset_name]["gt"]]
		else:
			if train:
				input_data = ["gt", "img", "prev"]
				images = [self.images[dataset_name]["gt"], self.images[dataset_name]["img"], self.images[dataset_name]["gen_" + str(res_id - 1)]] 
		
			else:
				input_data = ["gt", "prev"]
				images = [self.images[dataset_name]["gt"], self.images[dataset_name]["gen_" + str(res_id - 1)]]


		# Create patch info file
		filenames = list(self.images[dataset_name]["gt"].keys())
		patch_size = self.real_patch_size[res_id]
		info = patch_info(filenames, self.images[dataset_name]["gt"][filenames[0]].shape[1:], patch_size, self.stride[res_id], augmentation = self.augmentation[res_id])

		# Create dataset
		dataset = Dataset(info, images, input_data, self.model_patch_size[res_id])
		print("Dataset " + dataset_name + " of " + str(len(dataset)) + " images was created.")

		dataloader = DataLoader(dataset, batch_size = self.batch_size[res_id], shuffle = True, num_workers = 4)

		self.datasets[dataset_name][res_id] = {"dataset" : dataset, "dataloader" : dataloader, "info" : info}

		if write:
			for elt in input_data:
				self.write_patchwork_input(res_id, dataset_name, elt)
				


	def initialize_model(self, res_id):

		""" Initialize the generative models

		Keyword arguments:
		res_id -- scale id (int)
		"""


		print("Initializing model")

		if self.model_patch_size[res_id][0] == 64:
			nfe = (64, 128, 256, 512, 1024)
			nfd = (1024, 512, 256, 128, 64)
			ndf = (64, 128, 256)

		elif self.model_patch_size[res_id][0] == 128:
			nfe = (32, 64, 128, 256, 512)
			nfd = (512, 256, 128, 64, 32)
			ndf = (64, 128, 256, 256)

		elif self.model_patch_size[res_id][0] == 256:
			nfe = (16, 32, 64, 128, 256)
			nfd = (256, 128, 64, 32, 16)
			ndf = (64, 128, 256, 256, 256)
		else:
			print("Network resolution not implemented.")


		if self.model_type[res_id] == "RefinementcGAN":
			nc = self.n_gt_channels + self.n_img_channels
			no = self.n_img_channels
			ni = self.n_gt_channels + self.n_img_channels
			model = RefinementcGAN(nc, nfe, no, nfd, ni, ndf, self.lr[res_id], self.lbd[res_id])

		elif self.model_type[res_id] == "Refinementpix2pix":
			nc = self.n_gt_channels + self.n_img_channels
			no = self.n_img_channels
			ni = self.n_gt_channels + self.n_img_channels
			model = Refinementpix2pix(nc, nfe, no, nfd, ni, ndf, self.lr[res_id], self.lbd[res_id])

		elif self.model_type[res_id] == "RefinementcGAN_base":
			nc = self.n_gt_channels + self.n_img_channels
			no = self.n_img_channels
			ni = self.n_gt_channels + self.n_img_channels + self.n_img_channels
			model = RefinementcGAN_base(nc, nfe, no, nfd, ni, ndf, self.lr[res_id], self.lbd[res_id])
		
		elif self.model_type[res_id] == "pix2pix":
			nc = self.n_gt_channels
			no = self.n_img_channels
			ni = self.n_gt_channels + self.n_img_channels
			model = pix2pix(nc, nfe, no, nfd, ni, ndf, self.lr[res_id], self.lbd[res_id])

		elif self.model_type[res_id] == "ModeSeekingcGAN":
			
			nc = self.n_gt_channels
			no = self.n_img_channels
			ni = self.n_gt_channels + self.n_img_channels

			add_all = False
			model = ModeSeekingcGAN(nc, nfe, no, nfd, ni, ndf, self.lr[res_id], self.lbd[res_id], self.nz, add_all)
		
		elif self.model_type[res_id] == "cVAE":

			nfe = (32, 64, 128, 256)
			nfd = (256, 128, 64, 32)

			nint = 600
			k2 = 10e-5

			nl = self.n_gt_channels
			nc = self.n_img_channels
			no = self.n_img_channels
				
			model = cVAE(nc, no, nl, nfe, nfd, nint, self.nz, self.lr[res_id], k2 = k2) 

		else:
			print("Model type not recognized.")

		model.initialize_weights()

		self.models[res_id] = model


	def load_generated_image(self, res_id, dataset_name):

		""" Stitch patches to form the complete images 

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the image dataset (str) ex: "train"
		"""

		print("Loading generated images")
		
		self.images[dataset_name]["gen_" + str(res_id)] = {}

		# Store random z vectors used for the generation
		if dataset_name not in list(self.z.keys()):
			self.z[dataset_name] = {}

		def compute_weight(patch_shape, method):

			if method == "average":
				return np.zeros(patch_shape) + 1

			elif method == "weighted_average":
				def w(x,y, patch_shape):
					a = -(1/2) * pi
					b = (1/2) * pi
					wx = (b - a)*((x - 0)/(patch_shape[1] - 0)) + a
					wy = (b - a)*((y - 0)/(patch_shape[2] - 0)) + a
					return min([cos(wx), cos(wy)])

				weight = np.zeros(patch_shape)

				for k in range(patch_shape[0]):
					for i in range(patch_shape[1]):
						for j in range(patch_shape[2]):
							weight[k,i,j] = w(i+0.5,j+0.5, patch_shape)

				return torch.from_numpy(weight.astype(np.float32))

		# Generate images by stitching patches 
		info = self.datasets[dataset_name][res_id]["info"] 
		filenames = list(set(list(info["filename"]))) # Get file list from dataset 

		for f in filenames:

			# Get id in array for filename and no augmentation
			idx = list(np.where((info["filename"] == f) & (info["augmentation"] == "none"))[0])
			img_size = self.images[dataset_name]["gt"][f].shape 

			create_img = True
			for i in idx:

				# Get patch boundaries
				bb = [info["patch_xmin"].iloc[i], info["patch_xmax"].iloc[i], info["patch_ymin"].iloc[i], info["patch_ymax"].iloc[i]]
				patch_size = [int(bb[1] - bb[0]), int(bb[3] - bb[2])]

				# Generate patch
				single_dataloader = DataLoader(Subset(self.datasets[dataset_name][res_id]["dataset"], [i]), batch_size = 1)

				if self.model_type[res_id] == "ModeSeekingcGAN":

					patch, z_list = self.models[res_id].eval(single_dataloader, z_vectors=[])
					patch = patch[0]
					self.z[dataset_name][f] = z_list[0]

				elif self.model_type[res_id] == "cVAE":

					z = self.models[res_id].eval_encoder(single_dataloader)
					self.z[dataset_name][f] = z[0]

					# Randomly sampled vector
					#zs = torch.randn(1, self.models[res_id].nz)
					#patch = self.models[res_id].eval_decoder(single_dataloader, [zs])[0]

					# Sample using local gaussian
					sigma = 0.1
					random_file = random.choice(list(self.z[dataset_name].keys()))
					zs = sigma * torch.randn(1, self.models[res_id].nz) + self.z[dataset_name][random_file]
					patch = self.models[res_id].eval_decoder(single_dataloader, [zs])[0]

					# Repeat same 
					#z_fix = self.z[dataset_name][list(self.z[dataset_name].keys())[0]]
					#patch = self.models[res_id].eval_decoder(single_dataloader, [z_fix])[0]


				elif self.model_type[res_id] == "RefinementcGAN" or self.model_type[res_id] == "pix2pix" or self.model_type[res_id] == "Refinementpix2pix" or self.model_type[res_id] == "RefinementcGAN_base":
					patch = self.models[res_id].eval(single_dataloader)[0]

				else:
					print("Model type not recognized.")

				# Resize to original patch size
				tr = transforms.Resize(patch_size, max_size=None, antialias=True)
				patch = tr(patch[0,:,:,:])

				if create_img:
					img = torch.zeros([patch.shape[0], img_size[1], img_size[2]])
					weights = torch.zeros([patch.shape[0], img_size[1], img_size[2]])
					w = compute_weight(patch.shape, method = "weighted_average")
					create_img = False
				
				img[:, bb[0]:bb[1], bb[2]:bb[3]] = img[:, bb[0]:bb[1], bb[2]:bb[3]] + w * patch
				weights[:, bb[0]:bb[1], bb[2]:bb[3]] = weights[:, bb[0]:bb[1], bb[2]:bb[3]] + w

			weights[weights == 0.] = 1.
			img = img / weights

			self.images[dataset_name]["gen_"+ str(res_id)][f] = img


	def plot_loss(self, res_id):

		""" Plot training loss for current model

		Keyword arguments:
		res_id -- scale id (int)
		"""

		self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/training/")

		losses = self.models[res_id].losses
		for loss_name in losses.keys():

			f, ax = plt.subplots(1, 1)
			ax.plot(losses[loss_name])
			ax.set_xlabel("Iterations")
			ax.set_ylabel("Loss")
			ax.legend()
			f.savefig(self.output_folder + "/scale_" + str(res_id) + "/training/" + loss_name + "_loss.png",  bbox_inches='tight')

		plt.close()


	def write_dataset_file(self, dataset_name):

		""" Write the list of files in the dataset for replicability

		Keyword arguments:
		dataset_name -- name of the dataset (str) ex: "train"
		"""

		file = open(self.output_folder + "filenames_" + dataset_name + ".txt", "w")

		for f in self.images[dataset_name]["gt"].keys():
			 file.write(f + "\n")

		file.close() 


	def write_parameters(self):

		""" Write the model parameters """

		file = open(self.output_folder + "parameters.txt", "w")

		file.write("output folder : " + str(self.output_folder) + "\n")
		file.write("real patch size : " + str(self.real_patch_size) + "\n")
		file.write("model patch size : " + str(self.model_patch_size) + "\n")
		file.write("model type : " + str(self.model_type) + "\n")
		file.write("iterations : " + str(self.itrs) + "\n")
		file.write("augmentation : " + str(self.augmentation) + "\n")
		file.write("stride : " + str(self.stride) + "\n")
		file.write("lr : " + str(self.lr) + "\n")
		file.write("nz : " + str(self.nz) + "\n")
		file.write("lbd : " + str(self.lbd) + "\n")
		file.write("batch size : " + str(self.batch_size) + "\n")

		file.close() 



	def write_z_vectors(self, dataset_name):

		""" Write a txt file with the list of z vector used for each generated images in the dataset

		Keyword arguments:
		dataset_name -- name of the dataset (str) ex: "train"
		"""

		f = open(self.output_folder + dataset_name  + "_z.txt", "w")
		for k in self.z[dataset_name].keys():
			
			string = k
			for i in range(self.nz):
				string += "\t" + str(self.z[dataset_name][k].numpy()[0,i])
			
			f.write(string + "\n")
		f.close()


	def write_repeat(self, res_id, dataset_name, nb = 24, random_id = None):
		
		""" Write patchwork image by repeating the generation for a single input

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		nb -- number of images to generate
		"""

		self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/evaluation/")
		if random_id is None:
			random_id = random.randint(0, len(self.datasets[dataset_name][res_id]["dataset"]))
		
		single_dataloader = DataLoader(Subset(self.datasets[dataset_name][res_id]["dataset"], [random_id]*nb), batch_size = 64)
		
		if self.model_type[res_id] == "ModeSeekingcGAN":
			gen = self.models[res_id].eval(single_dataloader, z_vectors = [])[0][0]
		
		
		elif self.model_type[res_id] == "pix2pix" or self.model_type[res_id] == "Refinementpix2pix" or self.model_type[res_id] == "RefinementcGAN" or self.model_type[res_id] == "RefinementcGAN_base":
			gen = self.models[res_id].eval(single_dataloader)[0]

		elif self.model_type[res_id] == "cVAE":
			# Randomly sampled z vector
			z = torch.randn(nb, self.models[res_id].nz)
			gen = self.models[res_id].eval_decoder(single_dataloader, [z])[0]

		else:
			print("Model type not recognized.")

		for i in range(gen.shape[1]):
			img = vutils.make_grid(gen[:,i,:,:].unsqueeze(1), padding=2, normalize=True).cpu()
			img = np.transpose(img,(1,2,0))

			f, ax = plt.subplots(1, 1)
			ax.axis("off")
			ax.imshow(img, cmap = "gray")	
			f.savefig(self.output_folder + "/scale_" + str(res_id) + "/evaluation/" + dataset_name + "_channel_" + str(i) + "_repeat_" + str(random_id) + ".png", bbox_inches='tight')


	def write_interpolation(self, res_id, dataset_name, nb = 24):
		
		""" Write patchwork image by interpolating z for a single input

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		nb -- number of images to generate
		"""

		if self.model_type[res_id] != "ModeSeekingcGAN" and self.model_type[res_id] != "cVAE":
			print("Z interpolation can be performed only for ModeSeekingcGAN models.")
		else:

			self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/evaluation/")

			random_id = random.randint(0, len(self.datasets[dataset_name][res_id]["dataset"]))
			single_dataloader = DataLoader(Subset(self.datasets[dataset_name][res_id]["dataset"], [random_id]*nb), batch_size = 64)

			w = np.linspace(0,1,nb).tolist()
			if self.model_type[res_id] == "ModeSeekingcGAN":
			
				z0 = get_z_random(1, self.nz)[0]
				z1 = get_z_random(1, self.nz)[0]
				
				z_vectors = [torch.cat([torch.unsqueeze(torch.lerp(z0, z1, interp), 0) for interp in w],0)]
				gen = self.models[res_id].eval(single_dataloader, z_vectors = z_vectors)[0][0]

			else:

				w = np.linspace(0,1,nb).tolist()
				z0 = torch.randn(1, self.models[res_id].nz)
				z1 = torch.randn(1, self.models[res_id].nz)

				z_vectors = [torch.cat([torch.unsqueeze(torch.lerp(z0, z1, interp), 0) for interp in w],0)]
				gen = self.model.eval_decoder(single_dataloader, z_vectors)[0]

			for i in range(gen.shape[1]):
				img = vutils.make_grid(gen[:,i,:,:].unsqueeze(1), padding=2, normalize=True).cpu()
				img = np.transpose(img,(1,2,0))

				f, ax = plt.subplots(1, 1)
				ax.axis("off")
				ax.imshow(img, cmap = "gray")	
				f.savefig(self.output_folder + "/scale_" + str(res_id) + "/evaluation/" + dataset_name + "_channel_" + str(i) + "_interpolate_" + str(random_id) + ".png", bbox_inches='tight')


	def write_repeat_z(self, res_id, dataset_name, nb = 24): # TO GENERALIZE TO VAE
		
		""" Write patchwork image by repeating generation with a single vector z

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		nb -- number of images to generate
		"""

		self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/evaluation/")

		random_id = random.sample(np.arange(0, len(self.datasets[dataset_name][res_id]["dataset"])).tolist(), nb)
		single_dataloader = DataLoader(Subset(self.datasets[dataset_name][res_id]["dataset"], random_id), batch_size = 64)
		
		z = get_z_random(1, self.nz)

		z_vectors = [torch.cat([z for i in range(nb)],0)]

		if self.model_type[res_id] != "ModeSeekingcGAN":
			print("Z interpolation can be performed only for ModeSeekingcGAN models.")

		gen = self.models[res_id].eval(single_dataloader, z_vectors = z_vectors)[0][0]

		img = vutils.make_grid(gen, padding=2, normalize=True).cpu()
		img = np.transpose(img,(1,2,0))

		f, ax = plt.subplots(1, 1)
		ax.axis("off")
		ax.imshow(img, cmap = "gray")	
		f.savefig(self.output_folder + "/scale_" + str(res_id) + "/evaluation/" + dataset_name + "_repeat_z_" + str(z.numpy()[0]) + ".png", bbox_inches='tight')


	def write_patchwork_input(self, res_id, dataset_name, data_type):

		""" Write patchwork of input data from dataloader

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		data_type -- name of the image type (str) ex: "gt"
		"""

		self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/training/")
		nb = 64
		if len(self.datasets[dataset_name][res_id]["dataset"]) < nb:
			nb = len(self.datasets[dataset_name][res_id]["dataset"])

		single_dataloader = DataLoader(Subset(self.datasets[dataset_name][res_id]["dataset"], np.arange(0,nb).tolist()), batch_size = nb)
		for batch in single_dataloader:
			
			for i in range(batch[data_type].shape[1]):

				img = vutils.make_grid(batch[data_type][:,i,:,:].unsqueeze(1), padding=2, normalize=True).cpu()
				img = np.transpose(img,(1,2,0))

				f, ax = plt.subplots(1, 1)
				ax.axis("off")
				ax.imshow(img, cmap = "gray")	
				f.savefig(self.output_folder + "/scale_" + str(res_id) + "/training/" + dataset_name + "_"+ data_type + "_channel_" + str(i) + ".png", bbox_inches='tight')


	def write_patchwork_output(self, res_id, dataset_name):#, data_type = ["prev"]):

		""" Write patchwork of input data from dataloader

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		data_type -- name of the image type (str) ex: "gen_0"
		"""

		self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/training/")
		nb = 64
		if len(self.datasets[dataset_name][res_id]["dataset"]) < nb:
			nb = len(self.datasets[dataset_name][res_id]["dataset"])

		single_dataloader = DataLoader(Subset(self.datasets[dataset_name][res_id]["dataset"], np.arange(0,nb).tolist()), batch_size = nb)
		
		if self.model_type[res_id] == "ModeSeekingcGAN":
			gen, zvectors = self.models[res_id].eval(single_dataloader)
			gen = gen[0]
		
		#elif self.model_type[res_id] == "RefinementcGAN":
		#	gen = self.models[res_id].eval(single_dataloader, data_type)[0]
		
		elif self.model_type[res_id] == "pix2pix" or self.model_type[res_id] == "Refinementpix2pix" or self.model_type[res_id] == "RefinementcGAN" or self.model_type[res_id] == "RefinementcGAN_base":
			gen = self.models[res_id].eval(single_dataloader)[0]

		elif self.model_type[res_id] == "cVAE":
			z = torch.randn(nb, self.models[res_id].nz)
			gen = self.models[res_id].eval_decoder(single_dataloader, [z])[0]
		
		else:
			print("Model type not recognized.")

		current_epoch = self.models[res_id].current_epoch

		for i in range(gen.shape[1]):
			img = vutils.make_grid(gen[:,i,:,:].unsqueeze(1), padding=2, normalize=True).cpu()
			img = np.transpose(img,(1,2,0))

			f, ax = plt.subplots(1, 1)
			ax.axis("off")
			ax.imshow(img, cmap = "gray")	
			f.savefig(self.output_folder + "/scale_" + str(res_id) + "/training/" + dataset_name + "_channel_" + str(i) + "_epoch_" + str(current_epoch) + ".png", bbox_inches='tight')
		

	def write_images(self, res_id, dataset_name, image_set, nb = None, rgb = False):

		""" Write images 

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		image_set -- name of the image type (str) ex: "gen_0"
		nb -- number of images to write
		rgd -- True to write result as rgb images
		"""
		self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/" + dataset_name + "_" + image_set + "/", clear=True)
		
		
		filenames = list(self.images[dataset_name][image_set].keys())
		if nb is not None:
			filenames = filenames[:nb]

		for f in filenames:
			img = self.images[dataset_name][image_set][f].numpy()

			if rgb:
			
				for c in range(int(img.shape[0] / 3)):
					# Rescale between 0 and 255
					channel = np.zeros((img.shape[1], img.shape[2],3))
					channel[:,:,0] = img[c*3,:,:]
					channel[:,:,1] = img[c*3+1,:,:]
					channel[:,:,2] = img[c*3+2,:,:]

					channel = 255*((channel - channel.min())/(channel.max() - channel.min()))
					
					# Write to folder
					skimage.io.imsave(self.output_folder + "/scale_" + str(res_id) + "/" + dataset_name + "_" 
						+ image_set + "/" + f + "_channel_" + str(c) + ".png", channel.astype(np.uint8))

			else:
				for c in range(img.shape[0]):

					# Rescale between 0 and 255
					channel = 255*((img[c,:,:] - img[c,:,:].min())/(img[c,:,:].max() - img[c,:,:].min()))
					
					# Write to folder
					skimage.io.imsave(self.output_folder + "/scale_" + str(res_id) + "/" + dataset_name + "_" 
						+ image_set + "/" + f + "_channel_" + str(c) + ".png", channel.astype(np.uint8))
		

	def train(self, res_id, dataset_name, write = True):

		""" Train the model for one scale

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		"""

		print("Training " + self.model_type[res_id] + " model")

		epochs = int(self.itrs[res_id] / len(self.datasets[dataset_name][res_id]["dataloader"]))
		if epochs < 5:
			epochs = 5
		for e in range(epochs):
			self.models[res_id].train(self.datasets[dataset_name][res_id]["dataloader"])

			str_loss = "Epoch " + str(e) + " : "
			for loss_name in self.models[res_id].losses.keys():
				str_loss += loss_name + " = " + str(self.models[res_id].losses[loss_name][-1]) + " "
			print(str_loss)

			if epochs > 20:
				step = int(epochs // 20) 
			else:
				step = 1

			epochs_to_write = np.arange(0, epochs, step).astype(int)

			if write and (e in epochs_to_write):
				self.write_patchwork_output(res_id, dataset_name)

		if write:
			self.plot_loss(res_id)

	
	def run_training(self, res_id, dataset_name, write = True):

		""" Train the model for a given scale

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		"""

		print("\n### SCALE " + str(res_id) + " ###\n")

		self.create_dir(self.output_folder + "/scale_" + str(res_id) + "/training/", clear=True)

		# Create dataset
		self.create_dataset(res_id, dataset_name, train = True, write = write)


		# Initialize model
		self.initialize_model(res_id)

		# Train model
		self.train(res_id, dataset_name)

		# Write state_dict 
		self.save_state_dict(res_id, model_name = "model")

		# Write model
		#self.save_model(res_id, model_name = "model")

		# Load generated images
		self.load_generated_image(res_id, dataset_name)

		if write:
			# Write some images
			self.write_images(res_id, dataset_name, "gen_" + str(res_id), 10)
			#self.write_images(res_id, dataset_name, "gen_img_" + str(res_id), 10)
			

			# Write repeat and z vector
			if res_id == 0:
				self.write_repeat(res_id, dataset_name, nb = 24)
				#self.write_z_vectors(dataset_name)

	
		# Write structure
		#self.save(model_name = "model_" + dataset_name + "_scale_" + str(res_id))

		# Free memory
		if True:
			self.free_cuda_memory(res_id, dataset_name, erase_model = True)
			if res_id > 0:
				self.free_memory(dataset_name, "gen_" + str(res_id-1))
				#self.free_memory(dataset_name, "gen_img_" + str(res_id-1))

		
	def run_training_all(self, dataset_name, write = True):

		""" Train the models for all scales

		Keyword arguments:
		dataset_name -- name of the image dataset (str) ex: "train"
		"""

		for i in range(self.nb_res):
			self.run_training(i, dataset_name, write=write)
		

	def run_evaluation(self, res_id, dataset_name, write = True, free_memory = True):

		""" Evaluate the model for a given scale

		Keyword arguments:
		res_id -- scale id (int)
		dataset_name -- name of the dataset (str) ex: "train"
		"""
		
		print("\n### SCALE " + str(res_id) + " ###\n")

		# Create dataset
		self.create_dataset(res_id, dataset_name, train = False)
		self.write_patchwork_input(res_id, dataset_name, "gt")

		# Load generated images
		self.load_generated_image(res_id, dataset_name)

		if write:

			# Write some images
			self.write_images(res_id, dataset_name, "gen_" + str(res_id), 10)

			if res_id == 0:
				self.write_repeat(res_id, dataset_name, nb = 24)
				self.write_z_vectors(dataset_name)

		# Free memory
		if free_memory:
			self.free_cuda_memory(res_id, dataset_name, erase_model = True)
			if res_id > 0:
				self.free_memory(dataset_name, "gen_" + str(res_id-1))


	def run_evaluation_all(self, dataset_name, write = True):

		""" Train the models for all scales

		Keyword arguments:
		dataset_name -- name of the image dataset (str) ex: "train"
		"""

		for i in range(self.nb_res):
			self.run_evaluation(i, dataset_name, write = write)
		

	def correlation_gt(self, dataset_name, image_set, patch_size = 10, resize = True, write = True):

		""" Compute the mean pearson correlation between the input segmentation and the generated image """
		
		if write:
			file = open(self.output_folder + "correlation_label" + "_" + dataset_name + "_" + image_set + ".txt", "w")
			file.write("filename\tcorr\n")

		metric_array = []

		for f in self.images[dataset_name]["gt"].keys():
			corr_sum = 0
			count = 0

			gt = self.images[dataset_name]["gt"][f]#[list(self.images[dataset_name]["gt"].keys())[0]]
			img = self.images[dataset_name][image_set][f]

			if resize:
				downscale = transforms.Resize([int(gt.shape[1] / 2), int(gt.shape[2]/2)], max_size=None, antialias=True)
				gt = downscale(gt)
				img = downscale(img)

			for i in range(gt.shape[1]):
				for j in range(gt.shape[2]):
					if gt[0,i,j] > 0: # Within the neuron
						# Extract patch
						i1 = 0 if i - patch_size // 2 < 0 else i - patch_size // 2
						i2 = gt.shape[1]-1 if i + patch_size // 2 > gt.shape[1] - 1 else i + patch_size // 2
						j1 = 0 if j - patch_size // 2 < 0 else j - patch_size // 2
						j2 = gt.shape[2] - 1 if j + patch_size // 2 > gt.shape[2] - 1 else j + patch_size // 2

						C = torch.corrcoef(torch.cat((torch.flatten(gt[0, i1:i2, j1:j2]).unsqueeze(0), torch.flatten(img[0, i1:i2, j1:j2]).unsqueeze(0)), dim = 0))
					
						if not np.isnan(C.numpy()[0, 1]): 
							corr_sum += C.numpy()[0, 1] 
							count += 1 
			if count > 0:	
				mean_corr = corr_sum / count
				metric_array.append(mean_corr)
			else:
				mean_corr = np.nan
			if write:
				file.write(f + "\t" + str(mean_corr) + "\n")

		if write:
			metric_array = np.array(metric_array)
			file.write("mean" + "\t" + str(np.mean(metric_array)) + "\n")
			file.write("min" + "\t" + str(np.min(metric_array)) + "\n")
			file.write("max" + "\t" + str(np.max(metric_array)) + "\n")
			file.write("std" + "\t" + str(np.std(metric_array)) + "\n")

			file.close() 
	

	def correlation_repeat(self, dataset_name, image_set, resize = True, write = True):

		""" Compute the mean pearson correlation between the input segmentation and the generated image """

		if write:
			file = open(self.output_folder + "correlation_repeat" + "_" + dataset_name + "_" + image_set + ".txt", "w")
			file.write("filename\tvar\n")

		combined_tensors = {}
		for f in self.images[dataset_name]["gt"].keys():

			img = self.images[dataset_name][image_set][f]
			if resize :
				downscale = transforms.Resize([int(img.shape[1] / 2), int(img.shape[2]/2)], max_size=None, antialias=True)
				img = downscale(img)

			if f[:-5] + f[-4:] not in list(combined_tensors.keys()):
				combined_tensors[f[:-5] + f[-4:]] = torch.flatten(img).unsqueeze(0)
			else:
				combined_tensors[f[:-5] + f[-4:]] = torch.cat((combined_tensors[f[:-5] + f[-4:]], torch.flatten(img).unsqueeze(0)), dim = 0)
				
		metric_array = []
		for f in combined_tensors.keys():

			corr_sum = 0
			count = 0
			# Compute correlation
			C = torch.corrcoef(combined_tensors[f])
			print(C)

			for i in range(C.shape[0]):
				for j in range(C.shape[1]):	
					if i > j and not np.isnan(C.numpy()[i, j]): 
						corr_sum += C.numpy()[i, j] 
						count += 1 

			if count > 0:	
				mean_corr = corr_sum / count
				metric_array.append(mean_corr)
			else:
				mean_corr = np.nan

			if write:
				file.write(f + "\t" + str(mean_corr) + "\n")

		if write:
			metric_array = np.array(metric_array)
			file.write("mean" + "\t" + str(np.mean(metric_array)) + "\n")
			file.write("min" + "\t" + str(np.min(metric_array)) + "\n")
			file.write("max" + "\t" + str(np.max(metric_array)) + "\n")
			file.write("std" + "\t" + str(np.std(metric_array)) + "\n")

			file.close() 


	def correlation_scales(self, dataset_name, resize = True, write = True):

		""" Compute the mean pearson correlation between the input segmentation and the generated image """

		if write:
			file = open(self.output_folder + "correlation_scales" + "_" + dataset_name + ".txt", "w")
			file.write("filename\tvar\n")

		combined_tensors = {}
		for f in self.images[dataset_name]["gt"].keys():

			for i in range(len(self.itrs)):
				img = self.images[dataset_name]["gen_" + str(i)][f]
				if resize :
					downscale = transforms.Resize([int(img.shape[1] / 2), int(img.shape[2]/2)], max_size=None, antialias=True)
					img = downscale(img)

				if f[:-5] + f[-4:] not in list(combined_tensors.keys()):
					combined_tensors[f[:-5] + f[-4:]] = torch.flatten(img).unsqueeze(0)
				else:
					combined_tensors[f[:-5] + f[-4:]] = torch.cat((combined_tensors[f[:-5] + f[-4:]], torch.flatten(img).unsqueeze(0)), dim = 0)

		metric_array = []
		for f in combined_tensors.keys():

			corr_sum = 0
			count = 0
			# Compute correlation
			C = torch.corrcoef(combined_tensors[f])

			for i in range(C.shape[0]):
				for j in range(C.shape[1]):	
					if i > j and not np.isnan(C.numpy()[i, j]): 
						corr_sum += C.numpy()[i, j] 
						count += 1 

			if count > 0:	
				mean_corr = corr_sum / count
				metric_array.append(mean_corr)
			else:
				mean_corr = np.nan

			if write:
				file.write(f + "\t" + str(mean_corr) + "\n")

		if write:
			metric_array = np.array(metric_array)
			file.write("mean" + "\t" + str(np.mean(metric_array)) + "\n")
			file.write("min" + "\t" + str(np.min(metric_array)) + "\n")
			file.write("max" + "\t" + str(np.max(metric_array)) + "\n")
			file.write("std" + "\t" + str(np.std(metric_array)) + "\n")

			file.close() 


	def fid_metric(self, dataset_name, image_set1, image_set2,  write = True):
		
		if write:
			file = open(self.output_folder + "fid" + "_" + dataset_name + "_" + image_set1 + "_" + image_set2 + ".txt", "w")


		fid = FrechetInceptionDistance(feature=64, normalize =True)

		tensor1 = None
		tensor2 = None
		tr = transforms.Resize([299, 299], max_size=None, antialias=True)

		for f in self.images[dataset_name][image_set1].keys():

			img1 = tr(self.images[dataset_name][image_set1][f])
			img1 = (img1 - img1.min())/(img1.max() - img1.min())
			img1 = img1.unsqueeze(0).expand(1,3,299,299)

			if tensor1 is None:
				tensor1 = img1
			else:
				tensor1 = torch.cat((tensor1, img1), dim = 0)

		for f in self.images[dataset_name][image_set2].keys():

			img2 = tr(self.images[dataset_name][image_set2][f])
			img2 = (img2 - img2.min())/(img2.max() - img2.min())
			img2 = img2.unsqueeze(0).expand(1,3,299,299)

			if tensor2 is None:
				tensor2 = img2
			else:
				tensor2 = torch.cat((tensor2, img2), dim = 0)

		fid.update(tensor1, real=True)
		fid.update(tensor2, real=False)
		metric = fid.compute().numpy()
		if write:
			file.write(str(metric) +  "\n")
			file.close()

		

	def variance_metric(self, dataset_name, image_set, resize = True, write = True):

		""" Compute the variance between images generated from the same input """

		if write:
			file = open(self.output_folder + "variance" + "_" + dataset_name + "_" + image_set + ".txt", "w")
			file.write("filename\tvar\n")


		combined_tensors = {}
		for f in self.images[dataset_name]["gt"].keys():

			img = self.images[dataset_name][image_set][f]
			if resize :
				downscale = transforms.Resize([int(img.shape[1] / 2), int(img.shape[2]/2)], max_size=None, antialias=True)
				img = downscale(img)

			if f[:-5] + f[-4:] not in list(combined_tensors.keys()):
				combined_tensors[f[:-5] + f[-4:]] = img
			else:
				combined_tensors[f[:-5] + f[-4:]] = torch.cat((combined_tensors[f[:-5] + f[-4:]], img), dim = 0)

		# Compute variance 
		metric_array = []
		for f in combined_tensors.keys():
			var_tensor = torch.var(combined_tensors[f], dim = 0)
			mean_var = np.mean(torch.flatten(var_tensor).numpy())
			metric_array.append(mean_var)
			if write:
				file.write(f + "\t" + str(mean_var) + "\n")

		if write:
			metric_array = np.array(metric_array)
			file.write("mean" + "\t" + str(np.mean(metric_array)) + "\n")
			file.write("min" + "\t" + str(np.min(metric_array)) + "\n")
			file.write("max" + "\t" + str(np.max(metric_array)) + "\n")
			file.write("std" + "\t" + str(np.std(metric_array)) + "\n")

			file.close() 


	def free_cuda_memory(self, res_id, dataset_name, erase_model = False):

		# Delete model and dataset
		if erase_model:

			if self.model_type[res_id] == "cVAE":

				self.models[res_id].model.cpu()
				del self.models[res_id].model
				del self.models[res_id]

			else:

				self.models[res_id].generator.cpu()
				self.models[res_id].discriminator.cpu()
				del self.models[res_id].generator
				del self.models[res_id].discriminator
				del self.models[res_id]

			self.models.insert(res_id, None)

		#self.datasets[dataset_name][res_id]["dataset"].cpu()
		#self.datasets[dataset_name][res_id]["dataloader"].cpu()

		del self.datasets[dataset_name][res_id]["dataset"], self.datasets[dataset_name][res_id]["dataloader"]
		gc.collect()
		torch.cuda.empty_cache()


	def free_memory(self, dataset_name, image_set):

		del self.images[dataset_name][image_set]

		gc.collect()
		torch.cuda.empty_cache()


	def save(self, model_name = "model"):

		""" Save the current state of the multi-res model using pickle 

		Keyword arguments:
		model_name -- name to give to the pickle file without extension (str)
		"""

		with open(self.output_folder + model_name + ".pickle", 'wb') as file:
			pickle.dump(self, file)


	def save_model(self, res_id, model_name = "model"):

		""" Save the current state of the model using pickle 

		Keyword arguments:
		res_id -- scale id (int)
		model_name -- name to give to the pickle file without extension (str)
		"""
		
		with open(self.output_folder + "/scale_" + str(res_id) + "/" + model_name + ".pickle", 'wb') as file:
			pickle.dump(self.models[res_id], file)


	def save_state_dict(self, res_id, model_name = "model"):

		if self.model_type[res_id] == "cVAE":

			state_dict = self.models[res_id].get_current_state_dict()
			
			with open(self.output_folder + "/scale_" + str(res_id) + "/" + model_name + "_state_dict.pickle", 'wb') as file:
				pickle.dump(state_dict, file)

		else:

			state_dict_generator, state_dict_discriminator = self.models[res_id].get_current_state_dict()
			
			with open(self.output_folder + "/scale_" + str(res_id) + "/" + model_name + "_state_dict_generator.pickle", 'wb') as file:
				pickle.dump(state_dict_generator, file)

			with open(self.output_folder + "/scale_" + str(res_id) + "/" + model_name + "_state_dict_discriminator.pickle", 'wb') as file:
				pickle.dump(state_dict_discriminator, file)


	def load_z_vectors(self, dataset_name):

		if dataset_name not in list(self.z.keys()):
			self.z[dataset_name] = {}

		f = open(self.output_folder + dataset_name  + "_z.txt", "r")
		lines = f.readlines()

		for line in lines:
			line = line.replace("\n", "").split("\t")
			self.z[dataset_name][line[0]] = torch.tensor([float(line[i]) for i in range(1,len(line))]).unsqueeze(0)
			print(line[0], self.z[dataset_name][line[0]])

	
	def load_models(self):

		""" Load all models from folders"""

		for i in range(self.nb_res):
			
			try:
				with open(self.output_folder + "/scale_" + str(i) + "/model.pickle", 'rb') as file:
					self.models[i] = pickle.load(file)
			except: 
				print("Model for scale " + str(i) + " not found.")

	def load_state_dicts(self):

		""" Load all state_dicts from folders"""

		for i in range(self.nb_res):
			
			try:
				if self.model_type[i] == "cVAE":
					with open(self.output_folder + "/scale_" + str(i) + "/model_state_dict.pickle", 'rb') as file:
						state_dict = pickle.load(file)
					self.initialize_model(i)
					self.models[i].load_state_dict(state_dict)
				else:

					with open(self.output_folder + "/scale_" + str(i) + "/model_state_dict_generator.pickle", 'rb') as file:
						state_dict_generator = pickle.load(file)
					with open(self.output_folder + "/scale_" + str(i) + "/model_state_dict_discriminator.pickle", 'rb') as file:
						state_dict_discriminator = pickle.load(file)

					self.initialize_model(i)
					self.models[i].load_state_dict(state_dict_generator, state_dict_discriminator)

				print("Loading state dict for scale " + str(i) + ".")
			except: 
				print("State dict for scale " + str(i) + " not found.")


	def create_dir(self, path, clear = False):

		""" Creates a new directory in the output folder

		Keywords arguments:
		path -- path of the new directory (str)
		clear -- whether to clear an existing directory or not (bool)
		"""

		if clear:
			if os.path.exists(path):
				shutil.rmtree(path)
		os.makedirs(path, exist_ok=True)
















		


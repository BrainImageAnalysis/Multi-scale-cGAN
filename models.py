import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError
import numpy as np
from copy import deepcopy
import gc

### Refinement block ###

# Custom weights initialization 
def weights_init(m):

	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

class Block(nn.Module):

	def __init__(self, in_ch, out_ch, stride = 1):
		
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding = 1, stride = stride),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(True),
			nn.Conv2d(out_ch, out_ch, 3, padding = 1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(True)
			)
	
	def forward(self, input):
		return self.layers(input)


class Encoder(nn.Module):

	def __init__(self, nc, nfe):
		super().__init__()

		self.nfe = nfe
		self.input_block = Block(nc, nfe[0])
		self.blocks = nn.ModuleList([Block(nfe[i], nfe[i+1]) for i in range(len(nfe)-1)])
		self.max_pool = nn.MaxPool2d(2)
	
	def forward(self, input):
		ftrs = []

		out = self.input_block(input)
		ftrs.append(out)
		out = self.max_pool(out)
		
		for i in range(len(self.blocks)-1):
			
			out = self.blocks[i](out)
			ftrs.append(out)
			out = self.max_pool(out)
		

		out = self.blocks[-1](out)

		return out, ftrs


class Decoder(nn.Module):

	def __init__(self, no, nfd):
		super().__init__()

		self.nfd = nfd
		#self.upconvs = nn.ModuleList([nn.ConvTranspose2d(nfd[i], nfd[i+1], 2, 2) for i in range(len(nfd)-1)])
		self.upsamp = nn.Upsample(scale_factor = (2,2))
		self.upconvs = nn.ModuleList([nn.Conv2d(nfd[i], nfd[i+1], 3, stride = 1, padding = 1) for i in range(len(nfd)-1)])
		self.blocks = nn.ModuleList([Block(nfd[i], nfd[i+1]) for i in range(len(nfd)-1)]) 
		self.output_layer = nn.Sequential(nn.Conv2d(self.nfd[-1], no, 3, padding = 1), nn.Tanh())
		
	def forward(self, out, ftrs):
	
		for i in range(len(self.blocks)):
			out = self.upsamp(out)
			out = self.upconvs[i](out)
			out = torch.cat([out, ftrs[len(ftrs)-i-1]], dim=1)
			out = self.blocks[i](out)

		out = self.output_layer(out)

		return out
	

class Unet(nn.Module):

	def __init__(self, nc, nfe, no, nfd):
		super().__init__()
		self.encoder = Encoder(nc, nfe)
		self.decoder = Decoder(no, nfd)

	def forward(self, input):
		out, ftrs = self.encoder(input)
		out = self.decoder(out, ftrs)

		return out


class Discriminator(nn.Module):

	def __init__(self, nc = 3, ndf = (64, 128)):

		super(Discriminator, self).__init__()
		self.ndf = ndf

		self.init_layer = nn.Sequential(
			nn.Conv2d(nc, ndf[0], 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf[0]),
			nn.LeakyReLU(0.2, inplace=True))
		layers = []

		for i in range(len(ndf)-1):
			layers.append(nn.MaxPool2d(2, stride=2))
			
			layers.append(nn.Conv2d(ndf[i], ndf[i+1], 3, stride=1, padding = 1, bias=False))
			layers.append(nn.BatchNorm2d(ndf[i+1]))
			layers.append(nn.LeakyReLU(0.2, inplace=True))


		self.inter_layers = nn.Sequential(*layers)
			
		self.end_layer = nn.Sequential(
			nn.Conv2d(ndf[-1], 1, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		out = self.init_layer(input)
		out = self.inter_layers(out)
		out = self.end_layer(out)

		return out


class RefinementcGAN:

	def __init__(self, nc_u, nfe_u, no_u, nfd_u, no_d, ndf_d, lr, lbd):

		self.generator = Unet(nc_u, nfe_u, no_u, nfd_u)
		self.discriminator = Discriminator(no_d, ndf_d)

		self.param_g = [nc_u, nfe_u, no_u, nfd_u]
		self.param_d = [no_d, ndf_d]

		self.adversarial_loss = nn.BCELoss()
		self.l1_loss = nn.L1Loss()

		self.lr = lr 
		self.lbd = lbd
		self.current_epoch = 0
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}

		self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
		self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	

	def initialize_weights(self):

		self.generator.apply(weights_init)
		self.discriminator.apply(weights_init)

		self.generator.to(self.device)
		self.discriminator.to(self.device)

	def reset_network(self):

		self.generator = Unet(self.param_g[0], self.param_g[1], self.param_g[2], self.param_g[3])
		self.discriminator = Discriminator(self.param_d[0], self.param_d[1])
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		self.current_epoch = 0

	def train(self, dataloader):

		self.generator.train()
		self.discriminator.train()

		batch_losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		for batch in dataloader:

			# Train disciminator
			self.discriminator.zero_grad()

			gt = batch["gt"].to(self.device)
			prev = batch["prev"].to(self.device)
			target = batch["img"].to(self.device) 
	
			# Generate images
			generated = self.generator(torch.cat((gt, prev), dim=1))

			# Train the discriminator with fake images
			fake_pairs = torch.cat((gt, generated), dim=1)
			fake_pred_labels = self.discriminator(fake_pairs.detach())
			fake_labels = torch.full((gt.shape[0], 1, fake_pred_labels.shape[2], fake_pred_labels.shape[3]), 0., dtype=torch.float, device=self.device)
			fake_loss = self.adversarial_loss(fake_pred_labels, fake_labels)

			# Train the discriminator with real images
			real_pairs = torch.cat((gt, target), dim=1)
			
			real_pred_labels = self.discriminator(real_pairs)
			real_labels = torch.full((gt.shape[0], 1, real_pred_labels.shape[2], real_pred_labels.shape[3]), 1., dtype=torch.float, device=self.device)
			real_loss = self.adversarial_loss(real_pred_labels, real_labels)

			D_tot_loss = (fake_loss + real_loss)/2
			
			batch_losses["Discriminator"].append(D_tot_loss.item())

			D_tot_loss.backward()
			self.optimizer_d.step()

			# Train the generator
			self.generator.zero_grad()

			# Get L1 loss from prev
			low_res = nn.functional.interpolate(generated, scale_factor = 1./2, mode='bilinear') #res_down(generated)
			low_res = nn.functional.interpolate(low_res, scale_factor = 2, mode='bilinear') #res_up(low_res)
					
			l1l = self.l1_loss(low_res, prev)
			
			# Get adversarial loss from prev
			gen = torch.cat((gt, generated), dim=1)
			gen_pred_labels = self.discriminator(gen)

			adl = self.adversarial_loss(gen_pred_labels, real_labels)
				
			G_tot_loss = adl + self.lbd*l1l

			batch_losses["Generator_L1"].append(l1l.item())
			batch_losses["Generator_adv"].append(adl.item())

			G_tot_loss.backward()
			self.optimizer_g.step()

		self.losses["Discriminator"].append(sum(batch_losses["Discriminator"]) / len(batch_losses["Discriminator"]))
		self.losses["Generator_L1"].append(sum(batch_losses["Generator_L1"]) / len(batch_losses["Generator_L1"]))
		self.losses["Generator_adv"].append(sum(batch_losses["Generator_adv"]) / len(batch_losses["Generator_adv"]))
		
		self.current_epoch += 1

		

	def eval(self, dataloader):

		generated = []
		self.generator.eval()

		with torch.no_grad():
			for batch in dataloader:
				gt = batch["gt"].to(self.device)
				prev = batch["prev"].to(self.device)
				gen = self.generator(torch.cat((gt, prev), dim=1))
				generated.append(gen.detach().cpu())

		return generated


	def load_state_dict(self, state_dict_generator, state_dict_discriminator):

		""" Re load the model weights

		Keyword arguments:
		state_dict_generator -- torch state dict for the generator
		state_dict_discriminator -- torch state dict for the discriminator
		"""

		self.generator.load_state_dict(state_dict_generator)
		self.discriminator.load_state_dict(state_dict_discriminator)

	def get_current_state_dict(self):

		""" Returns the current state dictionaries """
		return self.generator.state_dict(), self.discriminator.state_dict()



class Refinementpix2pix:

	def __init__(self, nc_u, nfe_u, no_u, nfd_u, no_d, ndf_d, lr, lbd):

		self.generator = Unet(nc_u, nfe_u, no_u, nfd_u)
		self.discriminator = Discriminator(no_d, ndf_d)

		self.param_g = [nc_u, nfe_u, no_u, nfd_u]
		self.param_d = [no_d, ndf_d]

		self.adversarial_loss = nn.BCELoss()
		self.l1_loss = nn.L1Loss()

		self.lr = lr 
		self.lbd = lbd
		self.current_epoch = 0
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}

		self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
		self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def initialize_weights(self):

		self.generator.apply(weights_init)
		self.discriminator.apply(weights_init)

		self.generator.to(self.device)
		self.discriminator.to(self.device)

	def reset_network(self):

		self.generator = Unet(self.param_g[0], self.param_g[1], self.param_g[2], self.param_g[3])
		self.discriminator = Discriminator(self.param_d[0], self.param_d[1])
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		self.current_epoch = 0

	def train(self, dataloader):

		self.generator.train()
		self.discriminator.train()

		batch_losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		for batch in dataloader:

			# Train disciminator
			self.discriminator.zero_grad()

			gt = batch["gt"].to(self.device)
			prev = batch["prev"].to(self.device)
			target = batch["img"].to(self.device)
			
			# Generate images
			generated = self.generator(torch.cat((gt, prev), dim=1))

			# Train the discriminator with fake images
			fake_pairs = torch.cat((gt, generated), dim=1)
			fake_pred_labels = self.discriminator(fake_pairs.detach())
			fake_labels = torch.full((gt.shape[0], 1, fake_pred_labels.shape[2], fake_pred_labels.shape[3]), 0., dtype=torch.float, device=self.device)
			fake_loss = self.adversarial_loss(fake_pred_labels, fake_labels)

			# Train the discriminator with real images
			real_pairs = torch.cat((gt, target), dim=1)
			real_pred_labels = self.discriminator(real_pairs)
			real_labels = torch.full((gt.shape[0], 1, real_pred_labels.shape[2], real_pred_labels.shape[3]), 1., dtype=torch.float, device=self.device)
			real_loss = self.adversarial_loss(real_pred_labels, real_labels)

			D_tot_loss = (fake_loss + real_loss)/2

			batch_losses["Discriminator"].append(D_tot_loss.item())

			D_tot_loss.backward()
			self.optimizer_d.step()

			# Train the generator
			self.generator.zero_grad()

			l1l = self.l1_loss(generated, target)

			# Get adversarial loss from prev
			gen = torch.cat((gt, generated), dim=1)
			gen_pred_labels = self.discriminator(gen)
			adl = self.adversarial_loss(gen_pred_labels, real_labels)
				
			G_tot_loss = adl + self.lbd*l1l


			batch_losses["Generator_L1"].append(l1l.item())
			batch_losses["Generator_adv"].append(adl.item())

			G_tot_loss.backward()
			self.optimizer_g.step()

		self.losses["Discriminator"].append(sum(batch_losses["Discriminator"]) / len(batch_losses["Discriminator"]))
		self.losses["Generator_L1"].append(sum(batch_losses["Generator_L1"]) / len(batch_losses["Generator_L1"]))
		self.losses["Generator_adv"].append(sum(batch_losses["Generator_adv"]) / len(batch_losses["Generator_adv"]))

		self.current_epoch += 1


	def eval(self, dataloader):

		generated = []
		self.generator.eval()

		with torch.no_grad():
			for batch in dataloader:
				gt = batch["gt"].to(self.device)
				prev = batch["prev"].to(self.device)
				gen = self.generator(torch.cat((gt, prev), dim=1))
				generated.append(gen.detach().cpu())

		return generated



	def load_state_dict(self, state_dict_generator, state_dict_discriminator):

		""" Re load the model weights

		Keyword arguments:
		state_dict_generator -- torch state dict for the generator
		state_dict_discriminator -- torch state dict for the discriminator
		"""

		self.generator.load_state_dict(state_dict_generator)
		self.discriminator.load_state_dict(state_dict_discriminator)


	def get_current_state_dict(self):

		""" Returns the current state dictionaries """
		return self.generator.state_dict(), self.discriminator.state_dict()


class RefinementcGAN_base:

	def __init__(self, nc_u, nfe_u, no_u, nfd_u, no_d, ndf_d, lr, lbd):

		self.generator = Unet(nc_u, nfe_u, no_u, nfd_u)
		self.discriminator = Discriminator(no_d, ndf_d)

		self.param_g = [nc_u, nfe_u, no_u, nfd_u]
		self.param_d = [no_d, ndf_d]

		self.adversarial_loss = nn.BCELoss()
		self.l1_loss = nn.L1Loss()

		self.lr = lr 
		self.lbd = lbd
		self.current_epoch = 0
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}

		self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
		self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	

	def initialize_weights(self):

		self.generator.apply(weights_init)
		self.discriminator.apply(weights_init)

		self.generator.to(self.device)
		self.discriminator.to(self.device)

	def reset_network(self):

		self.generator = Unet(self.param_g[0], self.param_g[1], self.param_g[2], self.param_g[3])
		self.discriminator = Discriminator(self.param_d[0], self.param_d[1])
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		self.current_epoch = 0

	def train(self, dataloader):

		self.generator.train()
		self.discriminator.train()

		batch_losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		for batch in dataloader:

			# Train disciminator
			self.discriminator.zero_grad()

			gt = batch["gt"].to(self.device)
			prev = batch["prev"].to(self.device)
			target = batch["img"].to(self.device)
			
			# Generate images
			generated = self.generator(torch.cat((gt, prev), dim=1))

			# Train the discriminator with fake images
			
			fake_pairs = torch.cat((gt, prev, generated), dim=1)
			fake_pred_labels = self.discriminator(fake_pairs.detach())
			fake_labels = torch.full((gt.shape[0], 1, fake_pred_labels.shape[2], fake_pred_labels.shape[3]), 0., dtype=torch.float, device=self.device)
			fake_loss = self.adversarial_loss(fake_pred_labels, fake_labels)

			#low_res = nn.functional.interpolate(target, scale_factor = 1./2, mode='bilinear') #res_down(generated)
			#low_res = nn.functional.interpolate(low_res, scale_factor = 2, mode='bilinear')

			# Train the discriminator with real images
		
			real_pairs = torch.cat((gt, prev, target), dim=1)
			real_pred_labels = self.discriminator(real_pairs)
			real_labels = torch.full((gt.shape[0], 1, real_pred_labels.shape[2], real_pred_labels.shape[3]), 1., dtype=torch.float, device=self.device)
			real_loss = self.adversarial_loss(real_pred_labels, real_labels)

			D_tot_loss = (fake_loss + real_loss)/2

			batch_losses["Discriminator"].append(D_tot_loss.item())

			D_tot_loss.backward()
			self.optimizer_d.step()

			# Train the generator
			self.generator.zero_grad()

			l1l = self.l1_loss(generated, target)

			# Get adversarial loss from prev
			gen = torch.cat((gt, prev, generated), dim=1)
			gen_pred_labels = self.discriminator(gen)
			adl = self.adversarial_loss(gen_pred_labels, real_labels)
				
			G_tot_loss = adl + self.lbd*l1l


			batch_losses["Generator_L1"].append(l1l.item())
			batch_losses["Generator_adv"].append(adl.item())

			G_tot_loss.backward()
			self.optimizer_g.step()

		self.losses["Discriminator"].append(sum(batch_losses["Discriminator"]) / len(batch_losses["Discriminator"]))
		self.losses["Generator_L1"].append(sum(batch_losses["Generator_L1"]) / len(batch_losses["Generator_L1"]))
		self.losses["Generator_adv"].append(sum(batch_losses["Generator_adv"]) / len(batch_losses["Generator_adv"]))

		self.current_epoch += 1


	def eval(self, dataloader):

		generated = []
		self.generator.eval()

		with torch.no_grad():
			for batch in dataloader:
				gt = batch["gt"].to(self.device)
				prev = batch["prev"].to(self.device)
				gen = self.generator(torch.cat((gt, prev), dim=1))
				generated.append(gen.detach().cpu())

		return generated



	def load_state_dict(self, state_dict_generator, state_dict_discriminator):

		""" Re load the model weights

		Keyword arguments:
		state_dict_generator -- torch state dict for the generator
		state_dict_discriminator -- torch state dict for the discriminator
		"""

		self.generator.load_state_dict(state_dict_generator)
		self.discriminator.load_state_dict(state_dict_discriminator)


	def get_current_state_dict(self):

		""" Returns the current state dictionaries """
		return self.generator.state_dict(), self.discriminator.state_dict()


class pix2pix:

	def __init__(self, nc_u, nfe_u, no_u, nfd_u, no_d, ndf_d, lr, lbd):

		self.generator = Unet(nc_u, nfe_u, no_u, nfd_u)
		self.discriminator = Discriminator(no_d, ndf_d)

		self.param_g = [nc_u, nfe_u, no_u, nfd_u]
		self.param_d = [no_d, ndf_d]

		self.adversarial_loss = nn.BCELoss()
		self.l1_loss = nn.L1Loss()

		self.lr = lr 
		self.lbd = lbd
		self.current_epoch = 0
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}

		self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
		self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	

	def initialize_weights(self):

		self.generator.apply(weights_init)
		self.discriminator.apply(weights_init)

		self.generator.to(self.device)
		self.discriminator.to(self.device)

	def reset_network(self):

		self.generator = Unet(self.param_g[0], self.param_g[1], self.param_g[2], self.param_g[3])
		self.discriminator = Discriminator(self.param_d[0], self.param_d[1])
		self.losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		self.current_epoch = 0

	def train(self, dataloader):

		self.generator.train()
		self.discriminator.train()

		batch_losses = {"Discriminator":[], "Generator_L1":[], "Generator_adv":[]}
		for batch in dataloader:

			# Train disciminator
			self.discriminator.zero_grad()

			gt = batch["gt"].to(self.device)
			target = batch["img"].to(self.device) 
			
			# Generate images
			generated = self.generator(gt)

			# Train the discriminator with fake images
			fake_pairs = torch.cat((gt, generated), dim=1)
			fake_pred_labels = self.discriminator(fake_pairs.detach())
			fake_labels = torch.full((gt.shape[0], 1, fake_pred_labels.shape[2], fake_pred_labels.shape[3]), 0., dtype=torch.float, device=self.device)
			fake_loss = self.adversarial_loss(fake_pred_labels, fake_labels)

			# Train the discriminator with real images
			real_pairs = torch.cat((gt, target), dim=1)
			
			real_pred_labels = self.discriminator(real_pairs)
			real_labels = torch.full((gt.shape[0], 1, real_pred_labels.shape[2], real_pred_labels.shape[3]), 1., dtype=torch.float, device=self.device)
			real_loss = self.adversarial_loss(real_pred_labels, real_labels)

			D_tot_loss = (fake_loss + real_loss)/2

			batch_losses["Discriminator"].append(D_tot_loss.item())

			D_tot_loss.backward()
			self.optimizer_d.step()

			# Train the generator
			self.generator.zero_grad()

			# Get adversarial loss from prev
			gen = torch.cat((gt, generated), dim=1)
			gen_pred_labels = self.discriminator(gen)
			adl = self.adversarial_loss(gen_pred_labels, real_labels)

			l1l = self.l1_loss(generated, target)
				
			G_tot_loss = adl + self.lbd*l1l

			batch_losses["Generator_L1"].append(l1l.item())
			batch_losses["Generator_adv"].append(adl.item())

			G_tot_loss.backward()
			self.optimizer_g.step()

		self.losses["Discriminator"].append(sum(batch_losses["Discriminator"]) / len(batch_losses["Discriminator"]))
		self.losses["Generator_L1"].append(sum(batch_losses["Generator_L1"]) / len(batch_losses["Generator_L1"]))
		self.losses["Generator_adv"].append(sum(batch_losses["Generator_adv"]) / len(batch_losses["Generator_adv"]))

		self.current_epoch += 1


	def eval(self, dataloader):

		generated = []
		self.generator.eval()

		with torch.no_grad():
			for batch in dataloader:
				gt = batch["gt"].to(self.device)
				gen = self.generator(gt)
				generated.append(gen.detach().cpu())

		return generated



	def load_state_dict(self, state_dict_generator, state_dict_discriminator):

		""" Re load the model weights

		Keyword arguments:
		state_dict_generator -- torch state dict for the generator
		state_dict_discriminator -- torch state dict for the discriminator
		"""

		self.generator.load_state_dict(state_dict_generator)
		self.discriminator.load_state_dict(state_dict_discriminator)


	def get_current_state_dict(self):

		""" Returns the current state dictionaries """
		return self.generator.state_dict(), self.discriminator.state_dict()



### Mode seeking ###

class Block_z(nn.Module):

	def __init__(self, in_ch, out_ch, stride = 1):
		
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding = 1, stride = stride),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(True),
			nn.Conv2d(out_ch, out_ch, 3, padding = 1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(True)
			)
	
	def forward(self, x, z=None):

		if z is None:
			x_and_z = x
		else:
			z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3)) # expand the vectors to form a series of nz (64x64) images for each element in the batch 
			x_and_z = torch.cat([x, z_img], 1) # add as an additional channel to the input (batch size x (1 + nz) x imsizex x imsizey) 

		return self.layers(x_and_z)


class Encoder_z(nn.Module):

	def __init__(self, nc, nfe, nz, add_all):
		super().__init__()

		self.nz = nz
		self.add_all = add_all
		self.nfe = nfe
		self.input_block = Block_z(nc+nz, nfe[0])
		if add_all:
			self.blocks = nn.ModuleList([Block(nfe[i]+nz, nfe[i+1], stride = 1) for i in range(len(nfe)-1)])
		else:
			self.blocks = nn.ModuleList([Block(nfe[i], nfe[i+1], stride = 1) for i in range(len(nfe)-1)])
		self.max_pool = nn.MaxPool2d(2)
	
	def forward(self, x, z):


		z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3)) # expand the vectors to form a series of nz (64x64) images for each element in the batch 
		x = torch.cat([x, z_img], 1) # add as an additional channel to the input (batch size x (1 + nz) x imsizex x imsizey) 

		ftrs = []
		out = self.input_block(x)
		ftrs.append(out)
		out = self.max_pool(out)
		
		for i in range(len(self.blocks)-1):
			if self.add_all:
				out = self.blocks[i](out, z)
			else:
				out = self.blocks[i](out)

			ftrs.append(out)
			out = self.max_pool(out)
		
		if self.add_all:
			out = self.blocks[-1](out,z)
		else:
			out = self.blocks[-1](out)


		return out, ftrs


class Decoder_z(nn.Module):

	def __init__(self, no, nfd, nz, add_all):
		super().__init__()

		self.nz = nz
		self.add_all = add_all
		self.nfd = nfd

		#self.upsamp = nn.Upsample(scale_factor = (2,2))
		#self.upconvs = nn.ModuleList([nn.Conv2d(nfd[i], nfd[i+1], 3, stride = 1, padding = 1) for i in range(len(nfd)-1)])
		self.upconvs = nn.ModuleList([nn.ConvTranspose2d(nfd[i], nfd[i+1], 2, 2) for i in range(len(nfd)-1)])
		
		if add_all:
			self.blocks = nn.ModuleList([Block(nfd[i]+nz, nfd[i+1]) for i in range(len(nfd)-1)]) 
		else:
			self.blocks = nn.ModuleList([Block(nfd[i], nfd[i+1]) for i in range(len(nfd)-1)]) 

		self.output_layer = nn.Sequential(nn.Conv2d(self.nfd[-1], no, 3, padding = 1), nn.Tanh())
		
	def forward(self, out, ftrs, z):

		for i in range(len(self.blocks)):

			#out = self.upsamp(out)
			out = self.upconvs[i](out)
			out = torch.cat([out, ftrs[len(ftrs)-i-1]], dim=1)
			if self.add_all:
				out = self.blocks[i](out,z)
			else:
				out = self.blocks[i](out)

		out = self.output_layer(out)

		return out
	

class Unet_z(nn.Module):

	def __init__(self, nc, nfe, no, nfd, nz, add_all = False):
		super().__init__()
		self.encoder = Encoder_z(nc, nfe, nz, add_all)
		self.decoder = Decoder_z(no, nfd, nz, add_all)

	def forward(self, x, z):

		out, ftrs = self.encoder(x, z)
		out = self.decoder(out, ftrs, z)

		return out

		
def get_z_random(batch_size, nz, random_type='uni'):

	if random_type == 'uni':
		z = torch.rand(batch_size, nz) * 2.0 - 1.0
	elif random_type == 'gauss':
		z = torch.randn(batch_size, nz)
	return z


class ModeSeekingcGAN:

	def __init__(self, nc_u, nfe_u, no_u, nfd_u, no_d, ndf_d, lr, lbd, nz, add_all = False):

		self.generator = Unet_z(nc_u, nfe_u, no_u, nfd_u, nz, add_all)
		self.discriminator = Discriminator(no_d, ndf_d)

		self.param_g = [nc_u, nfe_u, no_u, nfd_u, nz, add_all]
		self.param_d = [no_d, ndf_d]

		self.adversarial_loss = nn.BCELoss()
		self.l1_loss = nn.L1Loss()

		self.nz = nz
		self.lr = lr 
		self.lbd = lbd
		self.current_epoch = 0
		self.losses = {"Discriminator":[], "Generator_adv":[], "Generator_ms":[]}

		self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
		self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	

	def initialize_weights(self):

		self.generator.apply(weights_init)
		self.discriminator.apply(weights_init)

		self.generator.to(self.device)
		self.discriminator.to(self.device)

	def reset_network(self):

		self.generator = Unet_z(self.param_g[0], self.param_g[1], self.param_g[2], self.param_g[3], self.param_g[4], self.param_g[5])
		self.discriminator = Discriminator(self.param_d[0], self.param_d[1])
		self.losses = {"Discriminator":[], "Generator_adv":[], "Generator_ms":[]}
		self.current_epoch = 0

	def train(self, dataloader):

		self.generator.train()
		self.discriminator.train()

		batch_losses = {"Discriminator":[], "Generator_adv":[], "Generator_ms":[]}
		for batch in dataloader:

			gt = batch["gt"].to(self.device)
			target = batch["img"].to(self.device) 

			# Get two set of random vectors
			z_random1 = get_z_random(gt.shape[0], self.nz)	
			z_random1 = z_random1.to(self.device)  
			z_random2 = get_z_random(gt.shape[0], self.nz).to(self.device)
			z_random2 = z_random2.to(self.device)

			fake_B = self.generator(torch.cat((gt, gt), 0), torch.cat((z_random1, z_random2), 0))
			fake_B_random1, fake_B_random2 = torch.split(fake_B, z_random1.size(0), dim=0)

			fake_B_random1_condition = torch.cat((gt, fake_B_random1), 1)
			fake_B_random2_condition = torch.cat((gt, fake_B_random2), 1)

			real_B_condition = torch.cat((gt, target), 1)

			# Train the generator
			self.generator.zero_grad()

			# G adversarial loss
			pred_fake1 = self.discriminator(fake_B_random1_condition)
			pred_fake2 = self.discriminator(fake_B_random2_condition)

			fake_labels = torch.full((gt.shape[0], 1, pred_fake1.shape[2], pred_fake1.shape[3]), 0., dtype=torch.float, device=self.device)
			real_labels = torch.full((gt.shape[0], 1, pred_fake2.shape[2], pred_fake2.shape[3]), 1., dtype=torch.float, device=self.device)

			loss_G_GAN = self.adversarial_loss(pred_fake1, real_labels) + self.adversarial_loss(pred_fake2, real_labels)


			# Mutual information loss
			lz = torch.mean(torch.abs(fake_B_random2 - fake_B_random1)) / torch.mean(torch.abs(z_random2 - z_random1))
			eps = 1 * 1e-5
			loss_lz = 1 / (lz + eps)

			loss_G = loss_G_GAN + self.lbd * loss_lz 
			batch_losses["Generator_adv"].append(loss_G_GAN.item())
			batch_losses["Generator_ms"].append(loss_lz.item())
		
			loss_G.backward()

			self.optimizer_g.step()

			# Train the discriminator
			self.discriminator.zero_grad()

			pred_fake1 = self.discriminator(fake_B_random1_condition.detach())
			pred_fake2 = self.discriminator(fake_B_random2_condition.detach())
			# real
			pred_real = self.discriminator(real_B_condition)
			loss_D_fake1 = self.adversarial_loss(pred_fake1, fake_labels)
			loss_D_fake2 = self.adversarial_loss(pred_fake2, fake_labels)
			loss_D_real= self.adversarial_loss(pred_real, real_labels)
			
			# Combined loss
			loss_D = loss_D_fake1 + loss_D_fake2 + 2*loss_D_real
			loss_D.backward()
			batch_losses["Discriminator"].append(loss_D.item())

			self.optimizer_d.step()
		

		self.losses["Discriminator"].append(sum(batch_losses["Discriminator"]) / len(batch_losses["Discriminator"]))
		self.losses["Generator_adv"].append(sum(batch_losses["Generator_adv"]) / len(batch_losses["Generator_adv"]))
		self.losses["Generator_ms"].append(sum(batch_losses["Generator_ms"]) / len(batch_losses["Generator_ms"]))

		self.current_epoch += 1

	

	def eval(self, dataloader, z_vectors = []):

		generated = []
		self.generator.eval()

		with torch.no_grad():
			c = 0
			for batch in dataloader:
				gt = batch["gt"].to(self.device)

				if len(z_vectors) == 0:
					z_vectors.append(get_z_random(gt.shape[0], self.nz))

				gen = self.generator(gt, z_vectors[c].to(self.device))
				z_vectors[c].detach().cpu()
				generated.append(gen.detach().cpu())
				c+=1

		return generated, z_vectors


	def load_state_dict(self, state_dict_generator, state_dict_discriminator):

		""" Re load the model weights

		Keyword arguments:
		state_dict_generator -- torch state dict for the generator
		state_dict_discriminator -- torch state dict for the discriminator
		"""

		self.generator.load_state_dict(state_dict_generator)
		self.discriminator.load_state_dict(state_dict_discriminator)

	def get_current_state_dict(self):

		""" Returns the current state dictionaries """
		return self.generator.state_dict(), self.discriminator.state_dict()



class Encoder_VAE(nn.Module):

	def __init__(self, nc, nf, nint, nz, mode = "VAE"):
		super(Encoder_VAE, self).__init__()

		self.nf = nf
		self.mode = mode
		self.activation = nn.ReLU()
		self.downsampling = nn.MaxPool2d(kernel_size = 2, stride = 2)

		self.init_layer = nn.Conv2d(nc, nf[0], kernel_size = 3, padding=1)
		self.conv_layers = nn.ModuleList([nn.Conv2d(nf[i], nf[i+1], kernel_size = 3, padding=1) for i in range(len(nf)-1)])
		

		self.dense_layers = nn.Sequential(
			nn.Flatten(start_dim=1),
			nn.Linear(nf[-1]*8*8, nint),
			nn.ReLU(),
			nn.Linear(nint, nz),
			)

		self.dense_layers_mu = nn.Sequential(
			nn.Flatten(start_dim=1),
			nn.Linear(nf[-1]*8*8, nint),
			nn.ReLU(),
			nn.Linear(nint, nz),
			)

		self.dense_layers_sigma = nn.Sequential(
			nn.Flatten(start_dim=1),
			nn.Linear(nf[-1]*8*8, nint),
			nn.ReLU(),
			nn.Linear(nint, nz),
			)

		
	def forward(self, x, l):

		#y = self.activation(self.init_layer(torch.cat((l, x), dim = 1)))
		y = self.activation(self.init_layer(x))

		for i in range(len(self.nf)-1):
			y = self.activation(self.conv_layers[i](self.downsampling(y)))

		if self.mode == "AE":
			y = self.dense_layers(y)
			return y

		elif self.mode == "VAE":

			mu = self.dense_layers_mu(y)
			sigma = torch.exp(self.dense_layers_sigma(y))
			y = mu + torch.randn_like(sigma) * sigma
		
			return mu, sigma, y

		

class Decoder_VAE(nn.Module):

	def __init__(self, nl, no, nf, nint, nz, upsampling_method = "Upsample", activation_function = "Tanh"):
		super(Decoder_VAE, self).__init__()

		self.nf = nf
		self.upsampling_method = upsampling_method
		self.activation = nn.ReLU()

		self.dense_layers = nn.Sequential(
			nn.Linear(nz, nint),
			nn.ReLU(),
			nn.Linear(nint, nf[0]*8*8),
			nn.ReLU(), 
			)

		if upsampling_method == "Upsample":
			self.upsampling = nn.Upsample(scale_factor = 2)
			self.conv_layers = nn.ModuleList([nn.Conv2d(nf[i]+nl, nf[i+1], kernel_size = 3, padding=1) for i in range(len(nf)-1)])
		else: 
			self.conv_layers = nn.ModuleList([nn.ConvTranspose2d(nf[i]+1, nf[i+1], kernel_size = 3, padding=1, stride = 2) for i in range(len(nf)-1)])


		if activation_function == "Tanh":
			self.out_layer = nn.Sequential(nn.Conv2d(nf[-1] + nl, no, kernel_size = 3, padding=1), nn.Tanh())
		else:
			self.out_layer = nn.Sequential(nn.Conv2d(nf[-1]+ nl, no, kernel_size = 3, padding=1), nn.Sigmoid())


	def forward(self, x, l):

		y = self.dense_layers(x)
		y = y.view((-1,self.nf[0],8,8))

		for i in range(len(self.nf) - 1):
			# Concatenate labels
			dwnl = nn.functional.interpolate(l, size = (y.shape[2], y.shape[3]), mode='bilinear')
			y = torch.cat((dwnl, y), dim=1)

			if self.upsampling_method == "Upsample":
				y = self.activation(self.conv_layers[i](self.upsampling(y)))
			else:
				y = self.activation(self.conv_layers[i](y))

		y = self.out_layer(torch.cat((l, y), dim = 1))

		return y
		
		 
class VAE(nn.Module):

	def __init__(self, nc, no, nl, nfe, nfd, nint, nz):
		# Initialize the layers
		super(VAE, self).__init__()
		self.encoder = Encoder_VAE(nc, nfe, nint, nz)
		self.decoder = Decoder_VAE(nl, no, nfd, nint, nz)

	def forward(self, x, l):
		mu, sigma, y = self.encoder(x, l)
		y = self.decoder(y, l)
		return mu, sigma, y

class AE(nn.Module):

	def __init__(self, nc, no, nfe, nfd, nint, nz):
		# Initialize the layers
		super(AE, self).__init__()
		self.encoder = Encoder_VAE(nc, nfe, nint, nz, mode = "AE")
		self.decoder = Decoder_VAE(no, nfd, nint, nz)

	def forward(self, x, l):
		y = self.encoder(x, l)
		y = self.decoder(y, l)
		return y


class cVAE:

	def __init__(self, nc, no, nl, nfe, nfd, nint, nz, lr, k1 = 1, k2 = 10e-5):

		self.model = VAE(nc, no, nl, nfe, nfd, nint, nz)
		self.k1 = k1
		self.k2 = k2
		self.nz = nz

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.mse_loss = torch.nn.MSELoss()
		
		self.lr = lr 
		self.current_epoch = 0
		self.losses = {"MSE_Loss":[], "KL_Loss" : []}

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

		self.best_epoch = 0
		self.best_loss = np.inf
		self.best_epoch_dict = deepcopy(self.model.state_dict())
		

	def initialize_weights(self):

		self.model.apply(weights_init)
		self.model.to(self.device)
		

	def train(self, dataloader):

		self.model.train()

		batch_losses = {"MSE_Loss":[], "KL_Loss" : []}

		for batch in dataloader:
			self.model.zero_grad()

			gt = batch["gt"].to(self.device)
			img = batch["img"].to(self.device)

			mu, sigma, out = self.model(img, gt)
			
			loss_1 = self.mse_loss(out, img)
			loss_2 = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
			loss = self.k1 * loss_1  + self.k2*loss_2

			
			batch_losses["MSE_Loss"].append(loss_1.item())
			batch_losses["KL_Loss"].append(loss_2.item())
			
			loss.backward()
			self.optimizer.step()

		self.losses["MSE_Loss"].append(sum(batch_losses["MSE_Loss"]) / len(batch_losses["MSE_Loss"]))
		self.losses["KL_Loss"].append(sum(batch_losses["KL_Loss"]) / len(batch_losses["KL_Loss"]))
		self.current_epoch += 1


	def eval_encoder(self, dataloader):
	
		output = []
		self.model.eval()

		with torch.no_grad():
			for batch in dataloader:
				gt = batch["gt"].to(self.device)
				img = batch["img"].to(self.device)

				mu, sigma, z = self.model.encoder(img, gt)
				output.append(z.detach().cpu())	

		return output

	def eval_decoder(self, dataloader, z_vectors):
	
		output = []
		self.model.eval()
		k = 0
		with torch.no_grad():
			for batch in dataloader:
				gt = batch["gt"].to(self.device)
				z = z_vectors[k].to(self.device)

				out = self.model.decoder(z, gt)
				output.append(out.detach().cpu())	
				k+=1

		return output


	def load_state_dict(self, state_dict):

		""" Re load the model weights

		Keyword arguments:
		state_dict -- torch state dict
		"""
		self.model.load_state_dict(state_dict)


	def get_current_state_dict(self):

		""" Returns the current state dictionaries """
		return self.model.state_dict()




	



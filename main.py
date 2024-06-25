from MultiResGenerator import MultiResGenerator


img_folder = "path_to_img_folder"
gt_folder = "path_to_label_folder"
output_folder = "path_to_output_folder"

# Patch-related hyperparameters
number_of_scales = 5
real_patch_size = [[1024, 1024], [512, 512], [256,256] , [128,128], [128,128]] # Size of the field of view to consider for each scale
model_patch_size = [[64, 64], [64, 64], [64, 64], [64,64], [128,128]] # Size of the patch (model input) for each scale
stride = [0, 256, 128, 64, 64] # Stride of the field of view used to draw patches 

augmentation = [["flip_vertical", "flip_horizontal", "rotation90", "rotation180"], ["none"], ["none"], ["none"], ["none"]] # Augmentation to apply for each scale
model_type = ["ModeSeekingcGAN", "RefinementcGAN", "RefinementcGAN", "RefinementcGAN", "RefinementcGAN"] # Type of model to use for each scale

# Other hyperparameters
lbd = [1, 10, 10, 10, 10] # Loss balance parameter lbd (depends on the type of model)
itrs = [3000, 3000, 6000, 6000, 6000] # Number of iteration to run for each scale
lr = [0.0002, 0.0002, 0.0002, 0.0002, 0.0002] # Learning rate to use for each scale
batch_size = [32, 32, 32, 32, 32] # Batch size for each scale
nz = 12 # Size of the random vector used for ModeSeekingcGAN


### TRAINING ALL ###

# Create the cascade model
model = MultiResGenerator(output_folder, len(real_patch_size), real_patch_size, model_patch_size, model_type, itrs, augmentation, stride, lr, lbd, nz, batch_size)
# Load the images and labels. In our experience, to generate non histogram equalized images, it is crucial to provide both the equalized and non-equalized version of the original images.
# The images and labels must have the same file names. Optional : file_list argument to load only part of the images contained in folder (given by a list of file names)
model.load_images([train_img_folder, train_img_folder, train_gt_folder], data_type = ["img","img","gt"], dataset_name = "train", equalize = [True, False, False], data_range = [-1,1])
model.run_training_all() # Train all scales one-by-one

model.write_images(number_of_scales-1, "train", "gen_" + str(number_of_scales-1)) # Write images generated at the last scales

"""
### TRAINING ONLY SOME STEPS ###

# Create the cascade model
model = MultiResGenerator(output_folder, len(real_patch_size), real_patch_size, model_patch_size, model_type, itrs, augmentation, stride, lr, lbd, nz, batch_size)
# Load the images and labels
model.load_images([train_img_folder, train_img_folder, train_gt_folder], data_type = ["img","img","gt"], dataset_name = "train", equalize = [True, False, False], data_range = [-1,1])
# Load all the previously trained models
model.load_state_dicts()

start_scale = 2 # Train from scale 2
for i in range(0,start_scale):
	model.load_generated_image(i, "train") # Load the images generated at previous scales
for i in range(start_scale, number_of_scales):
	model.run_training(i, "train", write = True)
	#model.write_images(i, "train", "gen_" + str(i)) # Write images generated at the intermediate scales

model.write_images(number_of_scales-1, "train", "gen_" + str(number_of_scales-1)) # Write images generated at the last scales


### EVALUATION ###

# Load the full cascade weights 
augmentation = [["none"], ["none"], ["none"], ["none"], ["none"]]
model = MultiResGenerator(output_folder, len(real_patch_size), real_patch_size, model_patch_size, model_type, itrs, augmentation, stride, lr, lbd, nz, batch_size)

model.load_images([train_gt_folder], ["gt"], "test", [False], data_range = [-1,1])
model.load_state_dicts()

for i in range(len(itrs)):
	model.run_evaluation(i, "test", write = False)

model.write_images(number_of_scales-1, "test", "gen_" + str(number_of_scales-1))

"""





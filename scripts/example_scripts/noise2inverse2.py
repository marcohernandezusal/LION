import matplotlib.pyplot as plt
import pathlib
import torch
from torch.utils.data import DataLoader, Subset
from LION.models.CNNs.MSDNets.MS_D import MS_D
from LION.models.CNNs.MSDNets.MS_D2 import MSD_Net
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.Noise2Inverse_solver2 import Noise2InverseSolver
from skimage.metrics import structural_similarity as ssim

def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())

# %%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/test_debugging/")
final_result_fname = "Noise2Inverse_MSD.pt"
checkpoint_fname = "Noise2Inverse_MSD_check_*.pt"
validation_fname = "Noise2Inverse_MSD_min_val.pt"
#
# %% Define experiment

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()

# smaller dataset for example. Remove this for full dataset
lidc_dataset = Subset(lidc_dataset, [i for i in range(100)])


# %% Define DataLoader

batch_size = 12
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=False)
lidc_test = DataLoader(experiment.get_testing_dataset(), batch_size, shuffle=False)

# %% Model
# Default model is already from the paper.
model = MS_D().to(device)

# %% Optimizer
train_param = LIONParameter()

# loss fn
loss_fn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 3
train_param.learning_rate = 10e-4
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
noise2inverse_parameters = Noise2InverseSolver.default_parameters()
solver = Noise2InverseSolver(
    model,
    optimiser,
    loss_fn,
    noise2inverse_parameters,
    True,
    experiment.geo,
    device=device
)

# set data
solver.set_training(lidc_dataloader)
solver.set_testing(lidc_test, my_ssim)

# set checkpointing procedure
solver.set_saving(savefolder, final_result_fname)
solver.set_loading(savefolder)
solver.set_checkpointing(checkpoint_fname, 2, load_checkpoint=True)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname)

# test

solver.test()

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")

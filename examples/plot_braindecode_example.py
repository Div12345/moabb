"""
===========================
Braindecode Example
===========================

This Example show how to apply a Braindecode model to a dataset on MOABB.

For the example a within session evaluation is used on the
Binary P300 example

"""
# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)

# Transformer to generate the data format required for Braindecode classifier

import matplotlib.pyplot as plt

# import mne
import seaborn as sns
import torch
from braindecode import EEGClassifier
from braindecode.datautil import create_from_X_y
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from sklearn.pipeline import Pipeline
from skorch.callbacks import LRScheduler

# from skorch.helper import predefined_split
#
# import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery


cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
# Extract number of chans and time steps from dataset
# n_chans = train_set[0][0].shape[0]
# input_window_samples = train_set[0][0].shape[1]

# hard-coded for now
n_chans = 26
input_window_samples = 1

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

# Send model to GPU
if cuda:
    model.cuda()


# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=None,  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device=device,
)

# 2 different methods
# 1 - Create the Braindecode_dataset with create_windows_from_events by activating return_epochs
# in evaluations definition

# requires BaseConcatDataset as input -
# pipe1 = Pipeline([('Braindecode_dataset', create_windows_from_events()),('Net',clf)])

# 2 - Do all preprocessing then simply just convert the numpy X and y to a Braindecode format
# without much of description, etc and directly give it to the net.
pipe = Pipeline([("Braindecode_dataset", create_from_X_y()), ("Net", clf)])

# Define Evaluation
paradigm = LeftRightImagery()
# Because this is being auto-generated we only use 2 subjects
dataset = BNCI2014001()
dataset.subject_list = dataset.subject_list[:2]
datasets = [dataset]
overwrite = False  # set to True if we want to overwrite cached results
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
)

results = evaluation.process(pipe)

print(results.head())

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results. We the first plot is a pointplot with the average
# performance of each pipeline across session and subjects.
# The second plot is a paired scatter plot. Each point representing the score
# of a single session. An algorithm will outperforms another is most of the
# points are in its quadrant.

fig, axes = plt.subplots(1, 2, figsize=[8, 4], sharey=True)

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=axes[0],
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=axes[0], zorder=1, palette="Set1")

axes[0].set_ylabel("ROC AUC")
axes[0].set_ylim(0.5, 1)

# paired plot
paired = results.pivot_table(
    values="score", columns="pipeline", index=["subject", "session"]
)
paired = paired.reset_index()

sns.regplot(data=paired, y="RG+LR", x="CSP+LDA", ax=axes[1], fit_reg=False)
axes[1].plot([0, 1], [0, 1], ls="--", c="k")
axes[1].set_xlim(0.5, 1)

plt.show()

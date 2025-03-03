
# ecog
ECOG Simulator

ECOG = Electro-corticogram.  A recording of brain electrical activity measured at the brain surface typically using electrode arrays.

The purpose of this repo is to create an artificial dataset that can be used as test cases through an AI/ML pipeline.

In particular, what motivated this exercise was the desire to explore 3D convolutional and 3D visual transformer type workflows.

So, in this repo, we specify a certain array size and spatial pattern of activations.

The implementation is simple object oriented programming of an individual ECOG recording.

Those ECOG recordings are aggregated into an ECOG Dataset.

The end goal is to run experiments through a series of neural network architectures.

The framework of choice is PyTorch with results displayed to TensorBoard.

This repo has some education value.

But lately, I have access to real experimental data.  See: speechBCI repo.

ecog.py - the ECOG Class
ecogds.py - the ECOG Dataset Class
genecog.py - this script generates simulated background + activation.
devmodel.py - the test workflow (PyTorch, TensorBoard)

# Simulated Electro-corticogram (ECOG) Data

The purpose of this repo is to generate artificial ECOG-data.

The use of an artificial ECOG dataset is to support development and testing of an ECOG AI/ML workflow.

The reason to do something like this is to have confidence in the code once real neural data is processed.

## Description

ECOG = Electro-corticogram.  A recording of brain electrical activity measured at the brain surface typically using electrode arrays.

What motivated this exercise was the desire to explore 3D convolutional and 3D visual transformer type workflows.

So, in this repo, we specify a certain array size and generate known spatial patterns of activations.

Here is a example of a frame from a time series:

![image](https://github.com/user-attachments/assets/fb9847f0-03c8-4471-b6ec-65f4ae12de2a)

Here is an example of the time series of an individual array node:

![image](https://github.com/user-attachments/assets/f66627c7-1e82-4d35-93c7-8224bbe0ef80)

Then when we run the workflows we know exactly what we expect to see and can gain some confidence that the code is correct.

The implementation is simple object oriented programming of an individual ECOG "recording".

Those ECOG recordings are aggregated into an ECOG Dataset.

The end goal is to run experiments through a series of neural network architectures.

The framework of choice is PyTorch with results displayed to TensorBoard.

This repo has some education value.

But lately, I have access to real experimental data. See: speechBCI repo.

## Getting Started

To generate data exercise the genecog.py script.

To do some simple neural network experiments exercise the devmodel.py script.

### Dependencies

* No special requirements beyond the imports listed in the scripts.
* This repo was developed and executed on AWS via VSS Code.

### Installing

* No special requirements beyond the imports listed in the scripts.
* This repo was developed and executed on AWS via VSS Code.

### Executing program

* See Getting Started above.

## Help

Send an email to: kgrajski@nurosci.com

## Authors

Kamil A. Grajski (kgrajski@nurosci.com)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* (https://gist.github.com/DomPizzie) for the ReadMe template.

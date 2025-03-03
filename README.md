# Simulated Electro-corticogram (ECOG) Data

The purpose of this repo is to generate artificial ECOG-data.
The use of an artificial ECOG dataset is to support develop and testing of an ECOG AI/ML workflow.

## Description

ECOG = Electro-corticogram.  A recording of brain electrical activity measured at the brain surface typically using electrode arrays.

What motivated this exercise was the desire to explore 3D convolutional and 3D visual transformer type workflows.

So, in this repo, we specify a certain array size and generate known spatial pattern of activations.

The implementation is simple object oriented programming of an individual ECOG "recording".

Those ECOG recordings are aggregated into an ECOG Dataset.

The end goal is to run experiments through a series of neural network architectures.

The framework of choice is PyTorch with results displayed to TensorBoard.

This repo has some education value.

But lately, I have access to real experimental data. See: speechBCI repo.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)

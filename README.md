# DeepFindET
Segmentation of CryoET Tomograms for Particle Picking with 3D Convolutional Networks 

## Introduction

As advancements in cryo-electron tomography (cryo-ET) continue to uncover the structure of proteins in situ, the ability to collect thousands of tomograms places significant demands on the capability to identify and extract proteins within crowded cellular environments. However, the complexity of these environments, combined with the inherent challenges of image acquisition, has made data mining and analysis of cryo-ET tomograms a persistent bottleneck in research.

To address these challenges, we developed a deep-learning based pipeline designed to streamline the training and execution of 3D autoencoder models for particle picking in cryo-ET experiments. Our framework provides researchers is built on top of [copick](https://github.com/copick/copick), a storage-agnostic API to easily access tomograms and segmentations across local or remote environments. Coupled with a [ChimeraX plugin](https://github.com/copick/chimerax-copick), users can easily annotate new tomograms and visualize segmentations or particles coordinates results preduced from DeepFindET. Together, these tools enable a more effective and precise identification of particles within the dense and intricate landscapes of cryo-ET datasets.

## Installation

DeepFindET is based on the Tensorflow package and has been tested on Linux (Debian 10). For optimal performance, especially during training, an Nvidia GPU with CUDA support is required. The current version has been tested on NVIDIA A100, H100, and A6000 GPUs. If you are using a different GPU, you may need to adjust certain parameters (e.g., patch and batch sizes) to fit your available memory.

Before installation, you may need a python environment on your machine. If this is not the case, we advise installing Anaconda.

(Optional) Before installation, we recommend first creating a virtual environment that will contain your DeepFinder installation:

`conda create --name deepfindet python=3.10`
`conda activate deepfindet`

Also, in order for Keras to work with your Nvidia GPU, you need to install CUDA. Once these steps have been achieved, the user should be able to run DeepFinder.

## Instructions for Use 

Detailed instructions for using DeepFindET are provided in the `tutorials/` folder. These tutorials include jupyter notebooks with comments that guide you through the usage of the package with a synthetic dataset available on the CryoET-Dataportal - including the available command-line options. For example, to run the target generation script:

`step1 create --config config_10007.json --target ribosome,9 --seg-target membrane --voxel-size 7.84 --tomogram-algorithm wbp --out-name remotetargets`

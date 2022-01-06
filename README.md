# Particle Tracking Using Recurrent Neural Networks
### Master Thesis by Claus Hofmann

This repository contains the code accompanying my master thesis. It is an adaptation of the approach from computer vision-based
tracking by Milan et al. [1] with improvements w.r.t. runtime and tracking performance.


## Running the code

### Prerequisites
- Python 3.7

### Setup

1) Clone the repository using `git`: \
`git clone <path/to/repository>`
2) Navigate in the cloned folder and install the dependencies (usage of virtual environment or anaconda environment is recommended): \
`pip install -r requirements.txt` 
3) (optional): If the model should run on GPU, additionally install GPU-dependencies: \
`pip install -r requirements-gpu.txt` \
Cuda and CudNN have to be set up on your machine for Tensorflow to work with GPU. This works best with Anaconda: \
`conda install cudatoolkit=10.1.243` \
`conda install cudnn=7.6`
4) Download the TrackML Data from Kaggle (`train1.zip` is sufficient) and extract it:
https://www.kaggle.com/c/trackml-particle-identification/data
5) Enter your local path to the data set in `load_data_luigi.py`: \
e.g. `detector_data_root = '/path/to/data/train_1'`

### Training a model

The repository contains pre-trained models. If you would like to train your own models, follow these steps:

1) Select a configuration for the network architecture from `network_configuration.py` or create a new one
2) Navigate to the `motion` directory
2) Specify the configuration in `train_multi_volume_predict.py` and `train_multi_volume_full.py` as `config_to_use`
3) Run `train_multi_volume_predict.py` - training only particle state prediction first allows for faster model convergence: \
`python motion/train_multi_volume_predict.py`
4) Run `train_multi_volume_full.py` for training a full model. The trained weights from step 3. will automatically be  loaded into the model:\
`python motion/train_multi_volume_full.py`

### Evaluation using TrackML Score

The code for model evaluation can be found in the notebook `motion/accuracy_evaluation.ipynb`. It contains the code for evaluating
the pre-defined model architectures.

### Visualisations

The code for the visualizations in my thesis as well as the evaluation of LSTM's accuracy for high particle densities
can be found in `visualisations/visualizations.ipynb`

## References

[1] Milan, A., Rezatofighi, H., Dick, A., and Reid, I.: Online multi-target tracking using recurrent neural networks. (2016)

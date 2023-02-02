# Neural Networks for Photonic Crystal Waveguides

Repository for summer project that followed on from my MPhys final year project *Designing Photonic Crystal Waveguides Using Neural Networks* (June 2022) supervised by [Dr Sebastian Schulz](https://github.com/sschulz365) at the University of St Andrews.

## Description of Code

### Dataset Generation

[simulations](code/simulations/): [cluster_generate.py](code/simulations/cluster_generate.py) uniformly samples the 7d design space and simulates the band structures of photonic crystal waveguides using [MIT Photonic Bands](https://mpb.readthedocs.io/en/latest/). Simulations are either 3D or 2D approximations using the effective period method depending on the chosen control file [simulations/MPB-control](code/simulations/MPB-control). As simulations are time consuming (taking minutes to several hours), they were run on the [Kennedy High Performance Computing Cluster](https://www.st-andrews.ac.uk/high-performance-computing/) in batches. The corresponding SLURM jobs are in [2D-jobs](code/simulations/2D-jobs/) and [3D-jobs](code/simulations/3D-jobs/).

[preprocessing](code/preprocessing/): [merge_csvs.py](code/preprocessing/merge_csvs.py) combines the output of batches of simulations into a single csv. [prepare_training_sets.py](code/preprocessing/prepare_training_sets.py) rescales features to the range $[0,1]$ and creates training, validation and test sets in the ratio 70:15:15.

### Neural Network Tuning

[NN-tuning](code/NN-tuning/): Neural Networks were built using Keras and trained and evaluated with [cluster_train.py](code/NN-tuning/cluster_train.py). Hyperparameters were set in SLURM jobs with the best performing hyperparameters for the 2D and 3D datasets being [2D-jobs/tune-7.job](code/NN-tuning/2D-jobs/tune-7.job) and [3D-jobs/tune-1.job](code/NN-tuning/3D-jobs/tune-1.job), respectively.

### Transfer Learning

[transfer-learning](code/transfer-learning/): The best 2D NN served as a pretrained model for the 3D dataset and this "transfer learning" process was done using [transfer_learning.py](code/transfer-learning/transfer_learning.py).

### Neural Network Evaluation

[learning-curve](code/learning-curve/): To assess the impact of training set size on NN performance, a learning curve was constructed. NNs were trained and tested using [learning-curve.py](code/learning-curve/learning-curve.py) on subsets of varying size in a number of jobs [learning-curve-jobs](code/learning-curve/learning-curve-jobs/).

[NN-speed-test](code/NN-speed-test) Measures the inference speed of trained NN.

## Related Projects

This project followed Anson Ho's [undergraduate disseration](https://github.com/ansonwhho/photonic-crystals-neural-networks) which aimed to predict figures of merit instead of band structures.

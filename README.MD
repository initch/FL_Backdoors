# Backdoors 101

Forked from [Github - backdoors101](https://github.com/ebagdasa/backdoors101)

* Startup Tensorboard: `tensorboard --logdir runs --port 19001`.

To run an experiment:
```shell script
python training.py --name mnist --params configs/mnist_params.yaml
```
 
Argument `name` specifies Tensorboard name and commit just records the
 commit id into a log file for reproducibility.


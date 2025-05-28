# Homework 3 Report

## Part 1: Backdoor Attack

**How my code works**:

Basicly, `train_backdoor.py` contains the entry to the entire program, and generate poisoned training data with `poison_cifar.py` and load poisoned data from `generate_clb_attack.py` in the case of a clean label backdoor attack.

Inside `poison_cifar.py`, `generate_trigger` returns the trigger pattern and the corresponding mask based on the chosen trigger type. Then we apply such trigger to the input images and use them to train and/or test(attack) the model. This process is defined as 
$$
\text{poisoned} = (1-\text{mask})\cdot\text{input} + \text{mask}\cdot(\text{pattern}\cdot\alpha + \text{input}\cdot(1-\alpha))
$$

We also modify the labels of the poisoned images to the target label, in `add_predefined_trigger_cifar`, we only keep those that are not originally of the target class for testing's sake. And in `add_trigger_cifar`, we keep all images so as to train the model.

In `generate_clb_attack.py`, we conduct PGD attack on the training images, but not to fool the model into misclassifying the images, but to degrade the usual features and make the model learn the trigger pattern. It is crucial not to change the original label, otherwise it is not "clean-label" anymore. Thus the perturbation is defined by causing the largest loss without really altering the classification result.

To be more specific, in the code, we generate 2 perturbations for each image, each perturbation is a sum of at most 100 steps of gradient descent, and such gradient descent stops when misclassification occurs. (At first there was an implementation error, the detection of misclassification was put at the end of PGD, but after fixing it the results matched the requirements. So there are 2 major logs, the first one is incomplete and incorrect.)

PS. `np.long` is deprecated and should be replaced with `np.int64` in class `CIFAR10CLB`.

The results are listed below, some notes:
1. all with 50 epochs of training
2. the poison rate is calculated as the poisoned training data over the total data, not over the amount of data with target label, thus the poison rate is far lower for clean-label attack.

| Trigger Type | Trigger Alpha | Train Accuracy | Poison rate | ASR    | ACC    |
| ------------ | ------------- | -------------- | ----------- | ------ | ------ |
| badnet       | 1.0           | 0.9765         | 0.05        | 1.0000 | 0.9276 |
| blend        | 0.2           | 0.9931         | 0.05        | 1.0000 | 0.9241 |
| clean-label  | 1.0           | 0.9934         | 0.01        | 0.8429 | 0.9335 |

(Record a small peculiarity here: the ASR for clean-label attack experiences a sharp jump at epoch 10, from around 0.05 to around 0.5. I did not found explanation to this phenomenon in hyperparameters or outer settings, thus we conjecture here that the model is only able to grasp the trigger pattern after already gaining a certain level of insight into the orginial features.)

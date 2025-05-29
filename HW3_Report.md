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

## Part 2: Backdoor Defense - Adversarial Network Pruning

<!-- PS. The code in `generate_masks.py`, function `load_state_dict` has two identical lines, peculiar. -->

I will paste the code here first.

```python
noise_params = noise_opt.param_groups[0]['params']
mask_params = mask_opt.param_groups[0]['params']

reset(model, rand_init=True)

for p_n in noise_params:
	p_n.requires_grad = True
for p_m in mask_params:
	p_m.requires_grad = False  

for _ in range(args.anp_steps):
	noise_opt.zero_grad()

	include_noise(model) 
	output_adv_step = model(images)
	loss_adv_maximize = -criterion(output_adv_step, labels)

	loss_adv_maximize.backward()
	sign_grad(model)
	noise_opt.step()

	with torch.no_grad():
		for p_n in noise_params:
			p_n.data.clamp_(-args.anp_eps, args.anp_eps)

for p_n in noise_params:
	p_n.requires_grad = False
for p_m in mask_params:
	p_m.requires_grad = True

include_noise(model) 
output_perturbed = model(images)
loss_perturbed = criterion(output_perturbed, labels)

exclude_noise(model)
output_clean = model(images)
loss_clean = criterion(output_clean, labels)

with torch.no_grad():
	pred = output_clean.data.max(1)[1]
	total_correct += pred.eq(labels.data.view_as(pred)).sum().item()

anp_loss = args.anp_alpha * loss_clean + (
	1 - args.anp_alpha) * loss_perturbed

total_loss += anp_loss.item()

mask_opt.zero_grad()
anp_loss.backward()
mask_opt.step() 
clip_mask(model) 
```

So first we optimize the noise parameters to perturb the weight and maximize adversarial loss. At this stage we freeze mask parameters and only optimize noise parameters. Because the optimizer minimizes the loss, we negate the adversarial loss to maximize it. Each step we take the sign of the gradient and clip the noise parameters to be within the range of `[-args.anp_eps, args.anp_eps]`.

Then once we obtained the perturbation, we freeze the noise parameters and optimize the mask parameters. We calculate the perturbed output and the clean output, including and excluding noise respectively, and compute the loss as a weighted sum of the two losses.

Finally, we update the mask parameters using the computed loss and clip the mask values to ensure they remain within valid bounds.

Then the **results of experiments**.

Basically I ran experiments with `anp_alpha` ranging from 0.01 to 0.99, and then tested the pruned network with `anp_alpha` from 0.01 to 0.27 and `threshold` from 0.20 to 0.35.
The results of the first experiment is stored in `mask_out/mask_output.log`, and the second one in `prune_out/prune_output.log`. 

1. When `anp_alpha` gets bigger, `PoisonAcc` gets bigger, but not very steadily. Randomness has a big influence on the results. And `CleanAcc` gets smaller very slightly overall, but the randomness fluctuation is also big compared to this trend. As for the final `ASR` and `ACC`, when we fluctuate `anp_alpha` and fix `threshold`, the results showed no recoginizable trend, but tend to flucuate with the quality of the mask generated. Generally, with large threshold, higher `anp_alpha` tends to have a bigger probability of low `ASR` (below 10%).
2. When `threshold` gets bigger, `PoisonAcc` decreases with sharp drops and plateaus, while `CleanAcc` decreases more steadily.
3. **Final experiment result**: From the experiment we found two classes of hyperparameters that yields the required results.
   + `anp_alpha = 0.14`, `threshold = 0.23`~`0.26`.
   + `anp_alpha = 0.15`, `threshold = 0.24`or `0.25`.
   + when `anp_alpha = 0.14`, `threshold = 0.26`, `ASR` = `0.0390`, `ACC` = `0.9227`. This mask has been stored at `mask_out/mask_values.txt`. Test result is saved at `prune_out/pruning_by_0.26.txt`.


import torch
from torch import nn
from torch.nn import Module
import copy
import math
import numpy as np
from collections import OrderedDict
from typing import List
from tasks.batch import Batch
from sklearn.cluster import DBSCAN
from tasks.fl.fl_task import FederatedLearningTask
from tasks.fl.mnistfed_task import MNISTFedTask
from tasks.fl.cifarfed_task import CifarFedTask
from tasks.fl.cifar100fed_task import Cifar100FedTask
from tasks.fl.tinyimagenetfed_task import TinyImageNetFedTask

from training import logger

# CRFL
def get_model_norm(model):
		squared_sum = 0
		for name, layer in model.named_parameters():
			squared_sum += torch.sum(torch.pow(layer.data, 2))
		return math.sqrt(squared_sum)


def clip_weight_norm(global_model, clip=14):
	total_norm = get_model_norm(global_model)
	max_norm = clip
	clip_coef = max_norm / (total_norm + 1e-6)
	current_norm = total_norm
	if total_norm > max_norm:
		for name, layer in global_model.named_parameters():
			layer.data.mul_(clip_coef)
		current_norm = get_model_norm
	logger.debug(f"[CRFL] total norm: {total_norm}, clipping norm:{clip}, current norm: {current_norm}")
	return current_norm


def add_differential_privacy_noise(global_model, sigma=0.001, cp=False):
	if not cp:
		for name, param in global_model.state_dict().items():
			if 'tracked' in name or 'running' in name:
				continue
			# print(name)
			dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
			param.add_(dp_noise)
	else:
		smoothed_model = copy.deepcopy(global_model)
		for name, param in smoothed_model.state_dict().items():
			if 'tracked' in name or 'running' in name:
				continue
			dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
			param.add_(dp_noise)
		return smoothed_model
	
# Deepsight
def deepsight_aggregate_global_model(task: FederatedLearningTask, local_models: List[Module]):
		def ensemble_cluster(neups, ddifs, biases):
			biases = np.array([bias.cpu().numpy() for bias in biases])
			#neups = np.array([neup.cpu().numpy() for neup in neups])
			#ddifs = np.array([ddif.cpu().detach().numpy() for ddif in ddifs])
			N = len(neups)
			# use bias to conduct DBSCAM
			# biases= np.array(biases)
			cosine_labels = DBSCAN(min_samples=3,metric='cosine').fit(biases).labels_
			print("cosine_cluster:{}".format(cosine_labels))
			# neups=np.array(neups)
			neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
			print("neup_cluster:{}".format(neup_labels))
			ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
			print("ddif_cluster:{}".format(ddif_labels))

			dists_from_cluster = np.zeros((N, N))
			for i in range(N):
				for j in range(i, N):
					dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
						neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j]))/3.0
					dists_from_cluster[j, i] = dists_from_cluster[i, j]
					
			print("dists_from_clusters:")
			print(dists_from_cluster)
			ensembled_labels = DBSCAN(min_samples=3,metric='precomputed').fit(dists_from_cluster).labels_

			return ensembled_labels
		
		global_weight = list(task.model.state_dict().values())[-2]
		global_bias = list(task.model.state_dict().values())[-1]

		biases = [(list(local_models[i].state_dict().values())[-1] - global_bias) for i in range(task.params.fl_no_models)]
		weights = [list(local_models[i].state_dict().values())[-2] for i in range(task.params.fl_no_models)]

		n_client = task.params.fl_no_models
		cosine_similarity_dists = np.array((n_client, n_client))
		neups = list()
		n_exceeds = list()

		# calculate neups
		sC_nn2 = 0
		for i in range(n_client):
			C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
			# print("C_nn:",C_nn)
			C_nn2 = C_nn * C_nn
			neups.append(C_nn2)
			sC_nn2 += C_nn2
			
			C_max = torch.max(C_nn2).item()
			threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
			n_exceed = torch.sum(C_nn2 > threshold).item()
			n_exceeds.append(n_exceed)
		# normalize
		neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
		print("n_exceeds:{}".format(n_exceeds))
		rand_input = None
		if isinstance(task, MNISTFedTask):
			rand_input = torch.randn((256, 1, 28, 28)).to(task.params.device)
		elif isinstance(task, CifarFedTask) or isinstance(task, Cifar100FedTask):
			# 256 can be replaced with smaller value
			rand_input = torch.randn((256, 3, 32, 32)).to(task.params.device)
		elif isinstance(task, TinyImageNetFedTask):
			# 256 can be replaced with smaller value
			rand_input = torch.randn((256, 3, 64, 64)).to(task.params.device)

		global_ddif = torch.mean(torch.softmax(task.model(rand_input), dim=1), dim=0)
		# print("global_ddif:{} {}".format(global_ddif.size(),global_ddif))
		client_ddifs = [torch.mean(torch.softmax(local_models[i](rand_input), dim=1), dim=0)/ global_ddif
						for i in range(task.params.fl_no_models)]
		client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])
		# print("client_ddifs:{}".format(client_ddifs[0]))

		# use n_exceed to label
		classification_boundary = np.median(np.array(n_exceeds)) / 2
		
		identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
		print("identified_mals:{}".format(identified_mals))
		clusters = ensemble_cluster(neups, client_ddifs, biases)
		print("ensemble clusters:{}".format(clusters))
		cluster_ids = np.unique(clusters)

		deleted_cluster_ids = list()
		for cluster_id in cluster_ids:
			n_mal = 0
			cluster_size = np.sum(cluster_id == clusters)
			for identified_mal, cluster in zip(identified_mals, clusters):
				if cluster == cluster_id and identified_mal:
					n_mal += 1
			print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))        
			if (n_mal / cluster_size) >= (1 / 3):
				deleted_cluster_ids.append(cluster_id)
		# print("deleted_clusters:",deleted_cluster_ids)
		temp_local_models = copy.deepcopy(local_models)
		for i in range(n_client-1, -1, -1):
			# print("cluster tag:",clusters[i])
			if clusters[i] in deleted_cluster_ids:
				local_models.pop(i)

		print("final clients length:{}".format(len(local_models)))
		if len(local_models)==0:
			local_models = temp_local_models
		task.aggregate_global_model(local_models)

# Robust LR
def sign_voting_aggregate_global_model(task: FederatedLearningTask, local_models):
	original_params = task.model.state_dict()

	# collect client updates
	updates = list()
	for i in range(len(local_models)):
		local_params = local_models[i].state_dict()
		update = OrderedDict()
		for layer, weight in local_params.items():
			update[layer] = local_params[layer] - original_params[layer]
		updates.append(update)

	# compute_total_update
	robust_lrs = compute_robustLR(task.model, updates)
	# count signs：
	flip_analysis = dict()
	for layer in robust_lrs.keys():
		n_flip = torch.sum(torch.gt(robust_lrs[layer], 0.0).int())
		n_unflip = torch.sum(torch.lt(robust_lrs[layer], 0.0).int())
		flip_analysis[layer] = [n_flip, n_unflip]

	# 注意只适用于每个客户端均分数据的情况
	for i, _ in enumerate(local_models):
		prop = float(1 / task.params.fl_total_participants)
		robust_lr_add_weights(original_params, robust_lrs, updates[i], prop)

	task.model.load_state_dict(original_params)
	logger.debug(f"[Robust LR] flip analysis: {flip_analysis}")
	return flip_analysis

def compute_robustLR(global_model, updates):
	layers = updates[0].keys()
	# signed_weights = OrderedDict()
	robust_lrs = OrderedDict()
	for layer, weight in global_model.state_dict().items():
		# signed_weights[layer] = torch.zeros_like(weight)
		robust_lrs[layer] = torch.zeros_like(weight)

	for layer in layers:
		for update in updates:
			robust_lrs[layer] += torch.sign(update[layer])
		robust_lrs[layer] = torch.abs(robust_lrs[layer])
		robust_lrs[layer][robust_lrs[layer] >= 2] = 1.0
		robust_lrs[layer][robust_lrs[layer] != 1.0] = -1.0
	return robust_lrs


def robust_lr_add_weights(original_params, robust_lrs, update, prop):
	for layer in original_params.keys():
		if 'running' in layer or 'tracked' in layer:
			original_params[layer] = original_params[layer] + update[layer] * prop
		else:
			original_params[layer] = original_params[layer] + update[layer] * prop * robust_lrs[layer]


# FedMD (Ensemble Distillation)
def ensemble_distillation(task: FederatedLearningTask, local_models):
	optimizer = task.make_optimizer(task.model)
	task.model.train()
	# TODO task.server_train_loader 给服务器分配数据 (2023.10.19)
	for i, data in enumerate(task.server_train_loader):
		batch = task.get_batch(i, data)
		batch = get_avg_logits(batch, local_models)
		optimizer.zero_grad()
		predicted_labels = task.model(batch.inputs)
		kl_div_loss = nn.KLDivLoss(reduction='batchmean')(predicted_labels.softmax(dim=-1).log(),
															batch.labels.softmax(dim=-1))
		kl_div_loss.backward()
		optimizer.step()

def get_avg_logits(self, batch: Batch, local_models) -> Batch:
	ensembled_batch = batch.clone()
	with torch.no_grad():
		total_logits = None
		for local_model in local_models:
			local_model.eval()
			logit = local_model(batch.inputs)
			total_logits = logit if total_logits is None else total_logits + logit
		avg_logit = total_logits / len(local_models)
		ensembled_batch.labels = avg_logit
	return ensembled_batch
import argparse
import datetime
import torch
import numpy as np
# import config
import torch.nn.functional as F
import logging
import yaml
import math
# from models.resnet_cifar import ResNet18
# from models.MnistNet import MnistNet
from hdbscan import HDBSCAN
# from image_helper import ImageHelper

logger = logging.getLogger("logger")
# 用到的层
fc_bias = "linear.bias"
fc_weight = "linear.weight"
feature_weight = "layer4.1.left.4.weight"  # "layer4.1.left.4.weight"

def check_ignored_weights(name) -> bool:
    for ignored in ['num_batches_tracked']:
        if ignored in name:
            return True

    return False

# 余弦距离
def cosine_distances(helper, update_dict, agent_name_keys):
    # helper.params.fl_no_models -> 20
    x = torch.zeros(helper.params.fl_no_models,
                    helper.params.fl_no_models, dtype=torch.float)

    for i in range(helper.params.fl_no_models):
        for j in range(helper.params.fl_no_models):
            update_i = update_dict[agent_name_keys[i]][0][fc_bias]
            update_j = update_dict[agent_name_keys[j]][0][fc_bias]
            x[i][j] = 1 - \
                torch.cosine_similarity(update_i, update_j, dim=0).item()
    # logger.info(x)
    return x.numpy().astype(np.double)


# 输出层神经元的更新能力，这里fc层 512->10
def Normalized_Update_Energy(helper, update_dict, agent_name_keys):
    # 第i个客户端的神经元更新能力
    NEUPs = torch.zeros(
        helper.params.fl_no_models, 10, dtype=torch.float)
    for i in range(len(agent_name_keys)):
        #NEUP = torch.zeros(10, dtype=torch.float).to(helper.params.device)
        # for j in range(10):
        #     neuron_update_j = abs(
        #         update_dict[agent_name_keys[i]][0]["fc2.bias"][j])
        #     neuron_update_j += (abs(update_dict[agent_name_keys[i]]
        #                         [0]["fc2.weight"][j])).sum()
        #     NEUP[j] = neuron_update_j
        #     sum += neuron_update_j**2
        neuron_update = abs(update_dict[agent_name_keys[i]][0][fc_bias]) + (
            abs(update_dict[agent_name_keys[i]][0][fc_weight])).sum(axis=1)
        sum = (neuron_update**2).sum()

        NEUPs[i] = neuron_update**2 / sum
    # logger.info(NEUPs)
    return NEUPs


# 根据NEUPs界定阈值,计算超越阈值数目
def threshold_exceeding(NEUPs):
    Threshold_Exceeding = []
    for i in range(len(NEUPs)):
        max_update = NEUPs[i].max()
        threshold = 0.1*max_update
        TE = 0
        for j in range(len(NEUPs[i])):
            if NEUPs[i][j] > threshold:
                TE += 1
        Threshold_Exceeding.append(TE)
    logger.info(Threshold_Exceeding)
    return Threshold_Exceeding


# 更新的模型和全局模型的差异性
def Division_Differences(helper, update_dict, local_model, global_model, agent_name_keys):
    DDIFs = torch.zeros(
        helper.params.fl_no_models, 10, dtype=torch.float)

    random_sample = torch.randn(100, 3, 32, 32).to(helper.params.device)
    # 将各客户端的更新转为各客户端的模型
    for i in range(len(agent_name_keys)):
        local_model.copy_params(global_model.state_dict())
        for name, data in local_model.state_dict().items():
            if check_ignored_weights(name):
                continue
            data.add_(update_dict[agent_name_keys[i]][0][name])

    # 第i个客户端的第j个神经元的差异性

        output_local = local_model(random_sample)
        output_local = F.log_softmax(output_local, dim=1)
        output_global = global_model(random_sample)
        output_global = F.log_softmax(output_global, dim=1)

        divide = output_local / output_global
        divide_sum = divide.sum(axis=0)
        divide_sum /= 100

        DDIFs[i] = divide_sum.detach()
    # logger.info(DDIFs)
    return DDIFs


# 利用TE进行分类
def adversary_classify(Threshold):
    boundary = np.median(Threshold)
    for i in range(len(Threshold)):
        if Threshold[i] <= boundary / 2:
            Threshold[i] = 1  # 1判为恶意客户端
        else:
            Threshold[i] = 0  # 0判为良性客户端
    return Threshold


# 根据特征聚类结果定义距离矩阵
def Distance_From_Cluster(helper, clusters):
    Distance = torch.zeros(helper.params.fl_no_models,
                           helper.params.fl_no_models, dtype=torch.float)
    clusters_label = clusters.labels_
    for i in range(helper.params.fl_no_models):
        for j in range(helper.params.fl_no_models):
            if clusters_label[i] == clusters_label[j]:
                Distance[i, j] = 0
            else:
                Distance[i, j] = 1
    return Distance


# 利用余弦距离，DDIFs，NEUPs进行聚类
def features_clustering(NEUPs, DDIFs, cosine_distances_matrix):
    # tsne1 = TSNE(n_components=2)
    # reduced_cosine_matrix = tsne1.fit_transform(cosine_distances_matrix)
    # metric="precomputed"
    clusters_1 = HDBSCAN(min_cluster_size=2, metric="precomputed")
    clusters_1.fit(cosine_distances_matrix)
    print("clustering cosine_distances: ", clusters_1.labels_)

    # tsne2 = TSNE(n_components=2)
    # reduced_NEUPs = tsne2.fit_transform(NEUPs)
    clusters_2 = HDBSCAN(min_cluster_size=2)
    clusters_2.fit(NEUPs)
    print("clustering NEUPs: ", clusters_2.labels_)

    # tsne3 = TSNE(n_components=2)
    # reduced_DDIFs = tsne3.fit_transform(DDIFs)
    clusters_3 = HDBSCAN(min_cluster_size=2)
    clusters_3.fit(DDIFs)
    print("clustering DDIFs: ", clusters_3.labels_)
    return clusters_1, clusters_2, clusters_3


# 根据距离得到最终聚类结果
def defense_clustering(distance):
    distance = distance.numpy().astype(np.double)
    #tsne = TSNE(n_components=2)
    #reduced_distance = tsne.fit_transform(distance)
    clusters = HDBSCAN(min_cluster_size=2, metric="precomputed")
    clusters.fit(distance)
    logger.info(clusters.labels_)
    return clusters


def filtering(helper, global_model, local_model, update_dict, agent_name_keys):
    # 余弦距离
    cosine_distances_matrix = cosine_distances(
        helper, update_dict, agent_name_keys)
    # 神经元更新能力
    NEUPs = Normalized_Update_Energy(helper, update_dict, agent_name_keys)
    # 阈值超越数目，越小越可能是恶意客户端，做成一个分类器
    TE = threshold_exceeding(NEUPs)
    # 各客户端训练数据差异性
    DDIFs = Division_Differences(
        helper, update_dict, local_model, global_model, agent_name_keys)
    # 利用TE进行分类, Lables 10*1   1为判定的恶意客户端，0为良性客户端
    Labels = adversary_classify(TE)
    print("Using TE to classify: ", Labels)
    # 利用NEUPs, DDIFs, cosine_distances_matrix进行聚类
    clusters_cosine, clusters_neups, clusters_ddifs = features_clustering(
        NEUPs, DDIFs, cosine_distances_matrix)

    # 利用聚类结果来计算距离
    distance_cosine = Distance_From_Cluster(helper, clusters_cosine)
    distance_neups = Distance_From_Cluster(helper, clusters_neups)
    distance_ddifs = Distance_From_Cluster(helper, clusters_ddifs)
    #print("Distance_Cosine: ", distance_cosine)
    #print("Distance_NEUPs: ", distance_neups)
    #print("Distance_DDIFs: ", distance_ddifs)
    distance = (distance_cosine + distance_ddifs + distance_neups) / 3
    #print("Distance_for_last_clustering: ", distance)
    # 得到最终聚类结果
    defense_clusters = defense_clustering(distance)
    results = defense_clusters.labels_

    # 利用聚类结果来进行判断
    clusters_dict = dict()
    for i in range(len(defense_clusters.labels_)):
        if results[i] in clusters_dict.keys():
            clusters_dict[results[i]].append(i)
        else:
            clusters_dict[results[i]] = [i]

    # 良性or恶意cluster的判断
    for cluster, model in clusters_dict.items():
        count = 0.0
        for i in range(len(model)):
            count += Labels[model[i]]
        count /= len(model)
        if count <= 0.33:
            model.append("benign")
        else:
            model.append("poisoned")

    logger.info(clusters_dict)
    # 选出良性客户端
    benign_name_keys = []
    for cluster, model_id in clusters_dict.items():
        if model_id[-1] == "benign":
            for i in range(len(model_id) - 1):
                benign_name_keys.append(agent_name_keys[model_id[i]])
    return clusters_dict, benign_name_keys

# 裁剪层
def clipping(update_dict, agent_name_keys):
    distance_model = []
    for i in range(len(agent_name_keys)):
        squared_sum = 0
        for name, data in update_dict[agent_name_keys[i]][0].items():
            squared_sum += torch.sum(torch.pow(data, 2))
        distance_model.append(math.sqrt(squared_sum))
    # print(distance_model)
    bound = np.median(distance_model)
    clip = bound / distance_model
    for i in range(len(clip)):
        clip[i] = clip[i] if clip[i] < 1.0 else 1.0
    
    # 限制每个客户端更新的大小
    for i in range(len(agent_name_keys)):
        for name, data in update_dict[agent_name_keys[i]][0].items():
            data.mul_(clip[i])
    return update_dict

def average_aggregation(helper, weight_accumulator, global_model, number_benign):
    # FedAvg
    for name, data in global_model.state_dict().items():
        update = weight_accumulator[name] * \
            (helper.params.eta / number_benign)
        if data.dtype == torch.int64:
            update = update.to(torch.int64)
        data.add_(update)
    return True

if __name__ == "__main__":
    # 测试  python defense_deepsight1.py --params utils/cifar_params_2.yaml
    parser = argparse.ArgumentParser(description="DBA Defense")
    parser.add_argument("--params", dest="params")
    args = parser.parse_args()
    with open(f"./{args.params}", "r") as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")

    if params_loaded["type"] == "cifar":
        helper = ImageHelper(current_time=current_time,
                             params=params_loaded, name="cifar")
    
    # CIFAR10
    global_model = ResNet18(name="Global", created_time="123")
    state_dict = torch.load(
        "saved_models/model_cifar_Feb.19_19.52.44/model_epoch_200.pt")
    global_model.load_state_dict(state_dict)
    global_model = global_model.to(helper.params.device)

    local_model = ResNet18(name="Local", created_time="1234")
    local_model = local_model.to(helper.params.device)

    # 选中的客户端下标
    agent_name_keys = [41, 73, 51, 74, 61, 75, 34, 52, 36, 56, 63, 46, 95, 48, 97, 72, 71, 21, 24, 92, 13, 79, 47, 87, 57, 6, 60, 89, 8, 69]
    
    # 梯度更新字典表  { 41：[{name:data,...}], 73:[{name:data,...}] ...}
    update_dict = np.load("update/cifar_May.12_22.08.52/update_dict_1.npy",
                          allow_pickle=True).item()
    
    # 调用过滤层
    logger.info("--------------------filtering----------------------")
    clusters_dict, benign_name_keys = filtering(
                helper, global_model, local_model, update_dict, agent_name_keys)

    logger.info("--------------------cliping----------------------")
    
    # 裁剪层 得到的梯度更新为裁剪过后的
    update_dict = clipping(update_dict, agent_name_keys)

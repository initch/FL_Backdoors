import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from models.resnet_cifar import resnet18
from tasks.task import Task


class Cifar100Task(Task):
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))

    def load_data(self):
        self.load_cifar_data()

    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        if self.params.poison_images:
            self.train_loader = self.remove_semantic_backdoors()
        else:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)
        # 2023.6.9 add - self.poisoned_test_loader, remove samples with target labels
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params.backdoor_label]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        self.poisoned_test_loader = DataLoader(self.test_dataset,
                                        batch_size=128,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                            range_no_id))

        return True

    def build_model(self) -> nn.Module:
        if self.params.pretrained:
            model = resnet18(pretrained=True)

            # model is pretrained on ImageNet changing classes to CIFAR
            model.fc = nn.Linear(512, len(self.classes))
        else:
            model = resnet18(pretrained=False, num_classes=100)
        return model

    def remove_semantic_backdoors(self):
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return: None
        """

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       sampler=SubsetRandomSampler(
                                           unpoisoned_images))

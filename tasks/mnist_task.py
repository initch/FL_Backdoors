import torch
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms

from models.lenet import LeNet5
from models.simple import CNN
from tasks.task import Task


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_mnist_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        self.train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        
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
        self.poisoned_test_loader = torch_data.DataLoader(self.test_dataset,
                                        batch_size=self.params.test_batch_size,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                            range_no_id))
        
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        return True
    
    def load_data(self):
        return self.load_mnist_data()

    def build_model(self):
        return CNN(num_classes=len(self.classes))

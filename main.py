from test_FakeRoastUtil_v2 import *
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
from copy import deepcopy
import code

SEEDS = [1,2,3,4,5]
SPARSITY = [.75, .5, 0.1, 0.05, 0.02, 0.01, 0.001, 0.0002, 0.0001]

MODELS = [models.vgg11_bn(pretrained=True)]#, models.resnet18(pretrained=True)]

## change final layers to adapt for tinyimagenet input
# MODELS[0].avgpool = torch.nn.AdaptiveAvgPool2d(1)
# MODELS[0].fc = torch.nn.Linear(MODELS[0].fc.in_features, 200)

# change final layers to adapt for CIFAR100 and CIFAR10 input
MODELS[0].classifier[6] = torch.nn.Linear(MODELS[0].classifier[6].in_features, 100)
# MODELS[0].classifier[6] = torch.nn.Linear(MODELS[0].classifier[6].in_features, 10)

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


TRAIN_DATASETS = [datasets.CIFAR100(root= './data', train=True, download=True, transform=transform)]#,
                #   datasets.ImageFolder(root="datasets/tiny-imagenet-200/" + "train", transform=transform),  
                #   datasets.CIFAR10(root= './data', train=True, download=True, transform=transform)]

TEST_DATASETS = [datasets.CIFAR100(root= './data', train=False, download=True, transform=transform)]#,
                #  datasets.ImageFolder(root="datasets/tiny-imagenet-200/" + "val", transform=transform),
                #  datasets.CIFAR10(root= './data', train=False, download=True, transform=transform)]

ITERS = 20
STEP = 80000
DATASET = "cifar100"

for init_seed in SEEDS:
    for model, train, test in zip(MODELS, TRAIN_DATASETS, TEST_DATASETS):
        for sparse in SPARSITY:
            roaster = test_mapper(deepcopy(model), sparse, init_seed)
            metrics = train_and_roast(roaster, ITERS, STEP, train, test)
            with open(f"results/{DATASET}/model:{model.__class__.__name__}_sparsity:{sparse}_seed:{init_seed}_iteration:{ITERS}_step:{STEP}_post:5.pkl", 'wb') as f:
                pickle.dump(metrics, f)


from test_FakeRoastUtil_v2 import *
import pickle
from copy import deepcopy
import sys
sys.path.append('Synaptic-Flow')
from Utils import load

SEEDS = [69, 420]
SPARSITY = [0.75, 0.5, 0.1, 0.05, 0.02, 0.01, 0.001, 0.0002, 0.0001]

DATASET = "cifar10"

input_shape, num_classes = load.dimension(DATASET)
MODELS = [load.model('pt_vgg11_bn', 'lottery')(input_shape, num_classes, True, True)]
TRAINLOADERS = [load.dataloader(DATASET, batch_size = 128, train = True, workers = 4)]
TESTLOADERS = [load.dataloader(DATASET, batch_size = 256, train = False, workers = 4)]

ITERS = 1
STEP = 2400

for init_seed in SEEDS:
    for model, train, test in zip(MODELS, TRAINLOADERS, TESTLOADERS):
        for sparse in SPARSITY:
            roaster = test_mapper(deepcopy(model), sparse, init_seed)
            metrics = train_and_roast(roaster, ITERS, STEP, train, test, DATASET, init_seed, sparse)
            with open(f"results/{DATASET}/pt/model:{model.__class__.__name__}_sparsity:{sparse}_seed:{init_seed}_iteration:{ITERS}_step:{STEP}_post:5.pkl", 'wb') as f:
                pickle.dump(metrics, f)


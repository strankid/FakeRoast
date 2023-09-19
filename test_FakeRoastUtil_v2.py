import torch
import numpy
import torchvision
import torchvision.transforms as transforms
from FakeRoastUtil_v2 import *
import pdb
from copy import deepcopy
import code



def train_model(model, train_dataset, test_dataset, metrics):
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    device = torch.device('cuda')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    metrics['accuracy'].append(test_model(model, test_dataset))
    print(metrics['accuracy'][-1])
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # print statistics
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        metrics['post_loss'].append(running_loss/len(trainloader))
        metrics['post_accuracy'].append(test_model(model, test_dataset))
        print(metrics['post_accuracy'][-1])
        running_loss = 0.0

    return metrics

def train_and_roast(roaster, iteration, step, train_dataset, test_dataset):
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    device = torch.device('cuda')
    model = roaster.model.to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    epoch = 0
    metrics = {'loss': [], 'accuracy': [], 'roasted': [], 'post_loss': [], 'post_accuracy': []}
    metrics['accuracy'].append(test_model(model, test_dataset))
    print(metrics['accuracy'][-1])
    while(True):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % iteration == 0:    # print every 2000 mini-batches
                metrics['loss'].append(running_loss/iteration)
                metrics['roasted'].append(roaster.k)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / iteration))
                running_loss = 0.0
                try:
                    roaster.update_k(step)    
                except:
                    print(model)
                    return train_model(model, train_dataset, test_dataset, metrics)
        epoch += 1
        metrics['accuracy'].append(test_model(model, test_dataset))
        print(metrics['accuracy'][-1])

def test_model(model, dataset):    
    testloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    model.eval()
    correct = 0
    total = 0

    device = torch.device("cuda")
    model = model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def test1(sparsity=0.5):
    model = torchvision.models.AlexNet()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model, module_limit_size=25000)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoaster(model, False, sparsity, verbose=NONE, module_limit_size=25000)
    model = roaster.process()
    print(model)
    af = get_module_params(model)
    print(int(roastable * sparsity), (af-fixed))
    assert(abs( int(roastable * sparsity) - (af - fixed)) < 5)

def test2(sparsity=0.5):
    model = torchvision.models.AlexNet()
    import pdb
    pdb.set_trace()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoaster(model, False, sparsity, verbose=NONE)
    model = roaster.process()
    print(model)
    af = get_module_params(model)
    print(int(roastable * sparsity), (af-fixed))
    assert(abs( int(roastable * sparsity) - (af - fixed)) < 5)


def test3(sparsity=0.5):
    model = torchvision.models.AlexNet()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE)
    model = roaster.process()
    print(model)


def test_mapper(model, sparsity=0.5, init_seed=1):
    mapper_args = { "mapper":"pareto", "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : init_seed }
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE, mapper_args=mapper_args, init_seed=init_seed)
    model = roaster.process()
    return roaster












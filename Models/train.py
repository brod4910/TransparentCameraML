import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import time
import shutil
from normalize import CastTensor, BiasNoise, TranslateImage, GaussianNoise, MaxNormalization, PeriodicShift
import torchvision.datasets as datasets
import numpy as np
from CocoDataset import CocoDataset

'''
    The main train function that combines model setup, training, testing, and extra rigor evaluation

    params:
        args: Arguments which contain the required information for model setup, training, and testing
        model: Model to be trained, tested, and evaluated
        device: device to send the data to while training, testing and evaluating (e.g cpu, cuda, cuda:0, cuda:1, etc)
        checkpoint: If the model is being reloaded then checkpoint will not be none, otherwise ignore checkpoint
'''
def train(args, model, device, checkpoint):

    # The transform to be appended for normal testing
    data_transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
        CastTensor()])

    print("\nImages resized to %d x %d" % (args.resize, args.resize))

    train_dataset = CocoDataset(args.train_dir, args.train_annFile, transform=data_transform)
    val_dataset = CocoDataset(args.val_dir, args.val_annFile, transform=data_transform)

    # Concatenate each dataset to create a joined dataset
    train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size= args.batch_size, 
    shuffle= True, 
    num_workers= args.num_processes,
    pin_memory= True
    )

    test_loader = (torch.utils.data.DataLoader(
    val_dataset,
    batch_size= args.batch_size, 
    shuffle= True, 
    num_workers= args.num_processes,
    pin_memory= True
    ))

    # set the optimizer depending on choice
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum= args.momentum, dampening=0, weight_decay= 0 if args.weight_decay is None else args.weight_decay, nesterov= False)
    elif args.optimizer == 'AdaG':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif args.optimizer == 'AdaD':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    elif args.optimizer == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay= 0 if args.weight_decay is None else args.weight_decay, momentum=args.momentum, centered=False)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # check for a checkpoint
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("\nUsing optimizer: %s" % (args.optimizer))

    # set the Loss function as CrossEntropy or MultiLabelMarginLoss
    if args.loss_fn == 'CELoss':
        criterion = torch.nn.CrossEntropyLoss().cuda() if device == "cuda" else torch.nn.CrossEntropyLoss()
    elif args.loss_fn == 'MMLoss':
        criterion = torch.nn.MultiLabelMarginLoss().cuda() if device == "cuda" else torch.nn.MultiMarginLoss()
    elif args.loss_fn == 'BCELoss'
        criterion = torch.nn.BCEWithLogitsLoss().cuda() if device == "cuda" else torch.nn.BCEWithLogitsLoss()

    # either take the minimum loss then reduce LR or take max of accuracy then reduce LR
    if args.plateau == 'loss':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', verbose= True, patience= 6)
    elif args.plateau == 'accuracy':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'max', verbose= True, patience= 6)

    print("\nReducing learning rate on %s plateau\n" % (args.plateau))

    best_prec1 = 0 if checkpoint is None else checkpoint['best_prec1']
    is_best = False

    del checkpoint

    total_time = time.clock()

    # train and validate the model accordingly
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_epoch(epoch, args, model, optimizer, criterion, train_loader, device)
        test_loss, accuracy = test_epoch(model, test_loader, device)

        if args.plateau == 'loss':
            scheduler.step(test_loss)
        elif args.plateau == 'accuracy':
            scheduler.step(accuracy)

        if accuracy > best_prec1:
            best_prec1 = accuracy
            is_best = True

        # save the model every epoch
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'time': time.clock() - total_time,
            'c_layers' : args.c_layers,
            'f_layers' : args.f_layers,
            'batch_size' : args.batch_size,
            'resize' : args.resize
        }, is_best)

        is_best = False
        
'''
    Evaluates the model depnding on if Bias, Shift or Gaussian are not None.
    The function evaluates the model using 4 different transforms, where shift can be either periodic shift, 
    translation or both. Each transform happens independently of each other, meaning that the model is 
    evaluated separately for each transform.

    params:
        model: Model to be evaluated
        device: device to send the data to while evaluating (e.g cpu, cuda, cuda:0, cuda:1, etc)
        args: Arguments which contain the required information for evaluation
        Bias: Contains bias to add constant noise to each image
        Shift: Shift constant to shift images by. (UNUSED: the value itself is 
                unused but needs to be present for shifting to occur)
        Gaussian: Std to be used during evaluation
'''
def evaluate_model(model, device, args, Bias= None, Shift= None, Gaussian= None):
    data_transforms = []

    if Bias is not None:
        data_transforms.append([BiasNoise(Bias)])
    if Shift is not None:
        if 't' in args.type_shift and 'p' in args.type_shift:
            shift_t = [TranslateImage(args.shift, 0, random= args.rigor),
                    PeriodicShift(args.shift, random= args.rigor)]
        elif 't' in args.type_shift:
            shift_t = [TranslateImage(args.shift, 0, random= args.rigor)]
        elif 'p' in args.type_shift:
            shift_t = [PeriodicShift(args.shift, random= args.rigor)]

        data_transforms.append(shift_t)
    if Gaussian is not None:
        data_transforms.append([GaussianNoise(Gaussian)])

    for d_transform in data_transforms:
        data_transform = [transforms.Resize((args.resize, args.resize)),
            MaxNormalization(0.0038910505836575876),
            *d_transform,
            CastTensor(),
            transforms.Normalize([157.11056947927852], [139.749640327443])
            ]

        test_dataset = LenslessDataset.LenslessDataset(
        csv_file= args.test_csv,
        root_dir= args.root_dir,
        bare_transform= data_transform
        )

        test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= args.batch_size,
        shuffle= True,
        num_workers= 4,
        pin_memory= True
        )

        test_epoch(model, test_loader, device)

'''
    Trains the given model for a single epoch of the training data.

    params:
        epoch: The epoch the model is currently on
        args: arguments which contain required information for training
        model: The model to be trained
        optimizer: The optimizer that will update the model on the backward pass
        criterion: The loss function that is used to calculate the model's predictions
        train_loader: The training loader for the dataset being trained on
        device: Device to send the data to while training (e.g cpu, cuda, cuda:0, cuda:1, etc)
'''
def train_epoch(epoch, args, model, optimizer, criterion, train_loader, device):
    model.train()

    total_train_loss = 0
    batch_loss = 0

    optimizer.zero_grad()                                   # Reset gradients tensors

    for batch_idx, (inputs, targets) in enumerate(train_loader):



        inputs, targets = inputs.to(device), targets.to(device)

        output = model(inputs)                     # Forward pass

        loss = criterion(output, targets)      # Compute loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader.dataset), loss.item()))

        del inputs, targets, loss, output

    total_train_loss /= len(train_loader.dataset)
    print('\nAveraged loss for training epoch: {:.4f}'.format(total_train_loss))

'''
    Tests the given model for a single epoch of the testing data.

    params:
        model: The model in question
        test_loader: The testing loader for the dataset being tested against
        device: Device to send the data to while testing (e.g cpu, cuda, cuda:0, cuda:1, etc)
'''
def test_epoch(model, test_loader, device):
    model.eval()
    test_loss = 0
    accuracy = 0
    correct = 0

    # validate the model over the test set and record no gradient history
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):

            input, target = input.to(device), target.to(device)

            output = model(input)
            # sum up batch loss
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='mean').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            del input, target, output

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

    return test_loss, accuracy

'''
    Checkpoints a model 

    params: 
        state: The model to save
        is_best: Boolean that contains true or false depending on if the model is better than the previously saved
        filename: name of the file to save the model
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

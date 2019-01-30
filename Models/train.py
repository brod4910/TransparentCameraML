import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import time
import shutil
import torchvision.datasets as datasets
import numpy as np
from . import CocoDataset
from . import normalize

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
        transforms.ToTensor()])

    print("\nImages resized to %d x %d" % (args.resize, args.resize))

    train_dataset = CocoDataset.CocoDataset(args.train_dir, args.train_annFile, transform=data_transform)
    val_dataset = CocoDataset.CocoDataset(args.val_dir, args.val_annFile, transform=data_transform)

    # Concatenate each dataset to create a joined dataset
    train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size= args.batch_size, 
    shuffle= True, 
    num_workers= args.num_processes,
    pin_memory= True
    )

    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size= args.batch_size, 
    shuffle= True, 
    num_workers= args.num_processes,
    pin_memory= True
    )

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
    elif args.loss_fn == 'BCELoss':
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
        test_loss, accuracy = test_epoch(model, val_loader, device, criterion)

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

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = torch.stack(labels, 1).type(torch.FloatTensor).to(device)

        output = model(inputs)                     # Forward pass

        loss = criterion(output, labels)      # Compute loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader.dataset), loss.item()))

        del inputs, labels, loss, output

    total_train_loss /= len(train_loader.dataset)
    print('\nAveraged loss for training epoch: {:.4f}'.format(total_train_loss))

'''
    Tests the given model for a single epoch of the testing data.

    params:
        model: The model in question
        test_loader: The testing loader for the dataset being tested against
        device: Device to send the data to while testing (e.g cpu, cuda, cuda:0, cuda:1, etc)
'''
def test_epoch(model, val_loader, device, criterion):
    model.eval()
    test_loss = 0
    accuracy = 0
    correct = 0

    # validate the model over the test set and record no gradient history
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = torch.stack(labels, 1).type(torch.FloatTensor).to(device)

            label_idxs = [[i for i, l in enumerate(label) if l == 1] for label in labels]

            corr_size = []

            for idxs in label_idxs:
                size = len(idxs)//2
                if size != 0:
                    corr_size.append(size)
                elif idxs and size == 0:
                    corr_size.append(1)
                else:
                    corr_size.append(0)

            output = model(inputs)
            # sum up batch loss
            preds = F.binary_cross_entropy_with_logits(output, labels, reduction='none')

            # get correctness of predictions
            for i, (pred, idxs, size) in enumerate(zip(preds, label_idxs, corr_size)):
                torch_idxs = torch.tensor(idxs).to(device)
                out = torch.index_select(pred, 0, torch_idxs)
                out = out.ge(.7).sum()

                # select all incorrect items and check for 
                # percentage of incorrectness
                incorrect_preds = select_items_except(idxs, pred)
                incorr_size = incorrect_preds.size()
                incorrect_sum = incorrect_preds.le(.5).sum()

                # predictions must be greater than half of the correct labels
                # and incorrectness must be less than half of incorrect labels
                # 
                if out.item() >= size and incorrect_sum.item() >= incorr_size[0] // 2:
                    correct += out.item()

                del torch_idxs
                
            # get the index of the max log-probability

            del inputs, labels, output

    test_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'
          .format(test_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

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

def select_items_except(idxs, pred):
    out = pred.clone()
    shift = 0

    for i in idxs:
        out = torch.cat([out[:(i - shift)], out[(i - shift)+1:]])
        shift += 1

    return out





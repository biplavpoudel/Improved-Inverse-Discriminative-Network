import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pynvml import *
import os
from tensorboardX import SummaryWriter
import time
from torch.utils.data import DataLoader
from dataset_process.dataset import SignDataLoader
from models.net import net
from loss import Loss
from utils import *
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("The device is {}".format(device))
args = parse_args()


def normalize(inputs):
    # Normalize to range [0, 1]
    return inputs / 255.0

def compute_accuracy(predicted, labels):
    for i in range(3):
        predicted[i][predicted[i] > 0.5] = 1
        predicted[i][predicted[i] <= 0.5] = 0
    predicted = predicted[0] + predicted[1] + predicted[2]
    
    predicted[predicted < 2] = 0
    predicted[predicted >= 2] = 1
    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy

def train():
    BATCH_SIZE = 32
    EPOCHS = args.n_epoch
    LEARNING_RATE = 0.001

    np.random.seed(0)
    torch.manual_seed(1)

    cuda = torch.cuda.is_available()

    train_set = SignDataLoader(train=True)
    test_set = SignDataLoader(train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2 * BATCH_SIZE, shuffle=False)

    # For testing if my dataloader is yielding Nan or inf
    # Spoiler: it isn't
    # image, labels = next(iter(train_loader))
    # print(torch.max(image[0][0]))
    # print(torch.max(labels))

    scaler = GradScaler(init_scale=2.**10, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)

    model = net()
    if cuda:
        model = model.cuda()

    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir='scalar')

    if cuda:
        criterion = criterion.cuda()
    iter_n = 0
    t = time.strftime("%m-%d-%H-%M", time.localtime())
    print("The size of train datasets with each batch of size 32 is", len(train_loader))
    best_valid_accuracy = 0

    for epoch in tqdm(range(1, EPOCHS + 1)):
        # Training is initiated
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            torch.cuda.empty_cache()

            optimizer.zero_grad()

            inputs, labels = inputs.cuda(), labels.cuda()

            # Normalize inputs
            inputs = normalize(inputs)

            with autocast(), torch.autograd.detect_anomaly(check_nan=True):
                predicted = model(inputs)
                # print(predicted)
                loss = criterion(*predicted, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            accuracy = compute_accuracy(predicted, labels)

            writer.add_scalar(t + '/train_loss', loss.item(), iter_n)
            writer.add_scalar(t + '/train_accuracy', accuracy, iter_n)

            # For Validation for each 100 iterations of training
            if (i+1) % 100 == 0:
                with torch.no_grad():
                    accuracies = []
                    for i_, (inputs_, labels_) in enumerate(tqdm(test_loader)):
                        labels_ = labels_.float()
                        if cuda:
                            inputs_, labels_ = inputs_.cuda(), labels_.cuda()
                        with autocast(), torch.autograd.detect_anomaly(check_nan=True):
                            predicted_ = model(inputs_)
                        accuracies.append(compute_accuracy(predicted_, labels_))
                    accuracy_ = sum(accuracies) / len(accuracies)
                    writer.add_scalar(t + '/test_accuracy', accuracy_, iter_n)
                print('\nValidation Accuracy:{:.3f}'.format(accuracy_))

                if accuracy_ >= best_valid_accuracy:
                    best_valid_accuracy = accuracy_
                    torch.save(model.state_dict(),
                               f'{args.model_prefix}_{accuracy_:.3%}.pt')

            iter_n += 1
            print('\nEpoch[{}/{}], iter {}, loss:{:.3f}, accuracy:{}'.format(epoch, EPOCHS, i, loss.item(), accuracy))

    writer.close()


if __name__ == '__main__':
    train()

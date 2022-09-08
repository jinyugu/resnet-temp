import os
# Model
import argparse
import torch
from net import ResNet50
from data_loader import trainloader,testloader
print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ResNet50()
net = net.to(device)

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('resume_num', type=int,
                    help='enter epoch_num to resume')

args = parser.parse_args()

#Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt %d.pth'% args.resume_num)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# best_acc = 0.0
# start_epoch = 0


import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 40 == 39:
            print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 40 == 39:
                print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pt')
        best_acc = acc


if __name__ == '__main__':
    start_epoch = args.resume_num  # start from epoch 0 or last checkpoint epoch

    for epoch in range(start_epoch+1, start_epoch + 20):
        train(epoch)
        test(epoch)
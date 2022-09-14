import os
# Model
import torch
from net import ResNet50
from data_loader import trainloader,testloader
print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ResNet50()
net = net.to(device)

# Load checkpoint.
# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./checkpoint/ckpt.pth')
# net.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']

best_acc = 0.0
start_epoch = 0


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
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
        # torch.save(state, './checkpoint/ckpt.pt')
        # torch.save(net,'./net/net%d.pt'%epoch)
        torch.save(net.state_dict(),'./state_dict/sd%d%f.pt'%(epoch,acc))
        best_acc = acc


def get_accuracy(model, x_orig, y_orig, bs=64, device=torch.device('cuda:0')):
    n_batches = x_orig.shape[0] // bs
    print("n_batches:",n_batches)
    acc = 0.
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(x)
        acc += (output.max(1)[1] == y).float().sum()
    print("accuracy on test:", acc)
    return (acc / x_orig.shape[0]).item()

if __name__ == '__main__':
    start_epoch = start_epoch  # start from epoch 0 or last checkpoint epoch
    x_val, y_val = next(iter(testloader))
    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')
    for epoch in range(start_epoch, start_epoch + 20):
        train(epoch)
        test(epoch)
        get_accuracy(model=net,x_orig=x_val,y_orig=y_val)

import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from CIFAR10_test_model import *

# Dataset preparation
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# Get data size
train_data_size = len(train_data)
test_data_size = len(test_data)

# Print data size
print('train data size:{}'.format(train_data_size))
print('test data size:{}'.format(test_data_size))

# Use DataLoader to load data from dataset
train_dataloader = DataLoader(train_data, shuffle=False, batch_size=64)
test_dataloader = DataLoader(test_data, shuffle=True, batch_size=64)

# Build neuron network
CIFAR10_model = CIFAR10_test()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(CIFAR10_model.parameters(), lr = learning_rate)

# Other model parameters
total_train_step = 0
total_test_step = 0
epoch = 60
subset_batch_num = 150

# Add Tensorboard
writer = SummaryWriter('CIFAR10_model_test_logs')

# Epoch
for step in range(epoch):
    print('----------Start {} round of training----------'.format(step+1))

    # Start training
    CIFAR10_model.train()
    current_batch_num = 0
    for data in train_dataloader:

        # Forward propagation
        imgs, targets = data
        output = CIFAR10_model(imgs)
        print(output.shape)
        print(targets.shape)

        # Compute loss, backward propagation and optimization
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get train step and loss
        current_batch_num += 1
        total_train_step += 1
        if total_train_step %10 == 0:
            print('Train step: {}, Loss: {}'.format(total_train_step, loss.item()))
            writer.add_scalar('Train loss', loss.item(), total_train_step)

        if current_batch_num  == subset_batch_num:
            break
        else:
            pass

    # Start testing in current epoch
    CIFAR10_model.eval()
    total_test_loss = 0
    total_accuracy = 0
    current_batch_num = 0
    with torch.no_grad():
        for data in test_dataloader:

            # Forward propagation and check loss
            imgs, targets = data
            output = CIFAR10_model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss
            current_batch_num += 1
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

            if current_batch_num == subset_batch_num:
                break
            else:
                pass

    # Get test loss and accuracy
    accuracy_percentage = total_accuracy/(64*current_batch_num)
    print ('Total test loss in current epoch: {}'.format(total_test_loss))
    #print ('Total accuracy in current epoch: {}'.format(total_accuracy))
    print ('Total accuracy in current epoch: {}'.format(accuracy_percentage))
    writer.add_scalar('Test loss', total_test_loss, total_test_step)
    #writer.add_scalar('Test accuracy', total_accuracy, total_test_step)
    writer.add_scalar('Test accuracy', accuracy_percentage, total_test_step)
    total_test_step += 1




# Visualization: one last test run, show test images, show test labels, show output labels
CIFAR10_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
current_batch_num = 0
print('----------Start final test----------')
for data in test_dataloader:

    # Show test images
    imgs, targets = data
    output = CIFAR10_model(imgs)
    writer.add_images('Test images', imgs, current_batch_num)

    # Show test labels
    targets_list = targets.tolist()
    label_list = ''
    for target in targets_list:
        label_list = label_list + CIFAR10_label[target] + ' / '
    writer.add_text('Test image labels', label_list, current_batch_num)

    # Show output labels
    preds_list = output.argmax(1).tolist()
    preds_label_list = ''
    for preds in preds_list:
        preds_label_list = preds_label_list + CIFAR10_label[preds] + ' / '
    writer.add_text('Preds image labels', preds_label_list, current_batch_num)

    current_batch_num += 1
    if  current_batch_num == subset_batch_num:
        break
    else:
        pass


writer.close()
torch.save(CIFAR10_model, 'CIFAR10_model_save')

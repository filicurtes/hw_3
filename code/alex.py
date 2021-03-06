import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from alexnet_dann import alexnet
#from pacs_manager import pacs
from torchvision.datasets import ImageFolder
import torchvision.models as models

print(torch.__version__)

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from plotting import plot 

import numpy as np
import datetime
import os, sys
import pandas as pd 

from matplotlib.pyplot import imshow, imsave
# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
print()

set=1

NUM_CLASSES = 7 # 7  
BATCH_SIZE = 256 # Higher batch sizes allows for larger learning rates changing the batch size, learning rate should change by the same factor 
LR = 1e-3           # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
NUM_EPOCHS = 30     # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20      # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down
LOG_FREQUENCY = 10
#alpha= 0.5
# Define transforms for training phase
train_transform = transforms.Compose([transforms.Resize(256),      # Resizes short size of the PIL image to 256
                                      transforms.CenterCrop(224),  # Crops a central square patch of the image
                                                                   # 224 because torchvision's AlexNet needs a 224x224 input!
                                                                   # Remember this when applying different transformations, otherwise you get an error
                                      transforms.ToTensor(),       # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalizes tensor with mean and standard deviation
])
# Define transforms for the evaluation phase
test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))                                    
])

#Prepare Pytorch train/test Datasets
photo_dir='PACS/photo'
art_dir='PACS/art_painting'
cartoon_dir='PACS/cartoon'
sketch_dir='PACS/sketch'

photo_dataset = ImageFolder(photo_dir,transform=train_transform)
art_dataset = ImageFolder(art_dir,transform=test_transform)
'''
cartoon_dataset = ImageFolder(train_dir,transform=train_transform)
sketch_dataset = ImageFolder(train_dir,transform=train_transform)
'''

source_dataset=photo_dataset
target_dataset=art_dataset

# Check dataset sizes
print('Source Dataset: {}'.format(len(source_dataset)))
print('Target Dataset: {}'.format(len(target_dataset)))
'''
train_indexes = np.arange(train_dataset.__len__())
test_indexes = np.arange(test_dataset.__len__())
print(train_indexes)
train_labels = np.empty(train_dataset.__len__(), dtype=int)
train_dom_labels = np.empty(train_dataset.__len__(),dtype=int)
test_labels= np.empty(test_dataset.__len__(),dtype= int)
test_dom_labels = np.empty(test_dataset.__len__(),dtype=int)

for index in train_indexes:
  train_labels[index] = train_dataset.__getitem__(index)[1]
print(train_labels)  
'''
#train_indexes, val_indexes, _, _ = train_test_split(train_indexes, train_labels, test_size=0.5, random_state=42, stratify=train_labels)
'''
val_dataset = Subset(train_dataset, val_indexes)
train_dataset = Subset(train_dataset, train_indexes)


train_classes = np.zeros(101)

for elem in train_dataset:
  train_classes[elem[1]] += 1

val_classes= np.zeros(101)

for elem in val_dataset:
  val_classes[elem[1]] += 1

print(train_classes)
ax=sns.barplot(x=np.linspace(0, 100, num=101),y=train_classes)
plt.savefig(f'{set}myfig.png')
'''


# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
source_dataloader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

net=models.alexnet(pretrained=True)

net.classifier[6]=nn.Linear(4096,NUM_CLASSES) 

# Define loss function
criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy

# Choose parameters to optimize
# To access a different set of parameters, you have to access submodules of AlexNet
# (nn.Module objects, like AlexNet, implement the Composite Pattern)
# e.g.: parameters of the fully connected layers: net.classifier.parameters()
# e.g.: parameters of the convolutional layers: look at alexnet's source code ;) 
parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet

# Define optimizer
# An optimizer updates the weights based on loss
# We use SGD with momentum
optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,nesterov=True)

# Define scheduler
# A scheduler dynamically changes learning rate
# The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


#TRAIN
# By default, everything is loaded to cpu
net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

cudnn.benchmark # Calling this optimizes runtime

current_step = 0

acc_train_list=[]
acc_test_list=[]
loss_train_list=[]
loss_test_list=[]

# Start iterating over the epochs
for epoch in range(NUM_EPOCHS):
  print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, scheduler.get_lr()))
  src_running_loss=0
  src_running_corrects=0
  src_cnt=0
  for images,labels in source_dataloader:
    # Bring data over the device of choice
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    net.train()

    optimizer.zero_grad()

    outputs = net(images)
    _, preds = torch.max(outputs.data, 1)
    # Compute loss based on output and ground truth
    loss = criterion(outputs,labels)
    src_running_loss += loss

    # Compute gradients for each layer and update weights
    loss.backward()  # backward pass: computes gradients

    optimizer.step() # update weights based on accumulated gradients

    current_step += 1
    src_cnt+=1
  src_running_corrects += torch.sum(preds == labels.data).data.item()
  src_accuracy = src_running_corrects / float(len(source_dataloader)*BATCH_SIZE) 
  acc_train_list.append(src_accuracy)
  net.train(False)

  avg_loss=src_running_loss/src_cnt
  avg_loss=avg_loss.data.cpu().numpy()
  print(f'Epoch {epoch+1} src_acc:{src_accuracy} src_loss: {avg_loss}')
  loss_train_list.append(avg_loss)
  # Step the scheduler
  scheduler.step() 


trg_running_corrects=0
trg_running_loss=0
for images,labels in target_dataloader:
  with torch.no_grad(): 
    images=images.to(DEVICE)
    labels=labels.to(DEVICE)

    outputs = net(images)
    # Get predictions
    _, preds = torch.max(outputs.data, 1)
    trg_running_corrects += torch.sum(preds == labels.data).data.item()
    loss = criterion(outputs,labels)
    trg_running_loss += loss

  trg_accuracy = trg_running_corrects / float(len(target_dataset))
  trg_loss=trg_running_loss/float(len(target_dataloader))
  trg_loss=trg_loss.data.cpu().numpy()

loss_test_list = np.full((NUM_EPOCHS),trg_loss,)
acc_test_list = np.full((NUM_EPOCHS),trg_accuracy)

#plot(loss_train_list,loss_test_list,'Loss','loss_source','loss_target','Loss vs Epochs',f'alex_loss_train_pre',set)
#plot(acc_train_list,acc_test_list,'Accuracy','accuracy_source','accuracy_target','Accuracy vs Epochs',f'alex_acc_train_pre',set)
cwd=os.getcwd()
os.chdir('files')
results={'acc_train':acc_train_list,'loss_train':loss_train_list,'acc_val':acc_test_list,'loss_val':loss_test_list}
results_df=pd.DataFrame(results)
results_df.to_csv(f'alex_results_{set}.csv',index=False)
os.chdir(cwd)


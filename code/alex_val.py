import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from alexnet_dann import alexnet
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from plotting import plot 
import numpy as np
import datetime
import os, sys
import pandas as pd 
from matplotlib.pyplot import imshow, imsave

print(torch.__version__)

# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
print()

set=41

NUM_CLASSES = 7 # 7  
BATCH_SIZE = 256 # Higher batch sizes allows for larger learning rates changing the batch size, learning rate should change by the same factor 
LR = 1e-3          # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
NUM_EPOCHS = 30     # Total number of training epochs (iterations over dataset)
STEP_SIZE = 25      # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down
LOG_FREQUENCY = 10
alpha= 0.1
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

photo_dataset = ImageFolder(photo_dir,transform = train_transform)
art_dataset = ImageFolder(art_dir,transform = test_transform)
sketch_dataset= ImageFolder(sketch_dir,transform = test_transform)
cartoon_dataset = ImageFolder(cartoon_dir, transform = test_transform)

source_dataset = photo_dataset
target_dataset = art_dataset
validation_1_dataset = sketch_dataset
validation_2_dataset = cartoon_dataset

# Check dataset sizes
print('Source Dataset: {}'.format(len(source_dataset)))
print('Target Dataset: {}'.format(len(target_dataset)))
print('Val Dataset 1: {}'.format(len(validation_1_dataset)))
print('Val Dataset 2: {}'.format(len(validation_2_dataset)))

# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
source_dataloader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

val_1_dataloader = DataLoader(validation_2_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#val_1_dataloader = DataLoader(validation_2_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

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
acc_val_1_list=[]
loss_val_1_list=[]

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
  src_avg_loss=src_running_loss / len(source_dataloader)
  src_avg_loss=src_avg_loss.data.item()
  loss_train_list.append(src_avg_loss)

  net.train(False)
  val_1_running_corrects=0
  val_1_running_loss=0
  for images,labels in val_1_dataloader:
    with torch.no_grad(): 

      images=images.to(DEVICE)
      labels=labels.to(DEVICE)

      outputs = net(images)
      # Get predictions
      _, preds = torch.max(outputs.data, 1)

      val_1_running_corrects += torch.sum(preds == labels.data).data.item()
      loss = criterion(outputs,labels)
      val_1_running_loss += loss

  val_1_accuracy = val_1_running_corrects / float(len(validation_1_dataset))
  val_1_loss=val_1_running_loss/float(len(val_1_dataloader))
  val_1_loss=val_1_loss.data.cpu().numpy()

  loss_val_1_list.append(val_1_loss)
  acc_val_1_list.append(val_1_accuracy)

  # Step the scheduler
  scheduler.step() 
'''
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

trg_class_acc_list=np.full((NUM_EPOCHS),trg_accuracy)
trg_class_loss_list=np.full((NUM_EPOCHS),trg_loss)
#plot(loss_train_list,loss_test_list,'Loss','loss_source','loss_target','Loss vs Epochs',f'alex_loss_train_pre',set)

#plot(acc_train_list,acc_test_list,'Accuracy','accuracy_source','accuracy_target','Accuracy vs Epochs',f'alex_acc_train_pre',set)
'''
cwd=os.getcwd()
os.chdir('files')
results={'acc_train':acc_train_list,'loss_train':loss_train_list,'acc_val_1':acc_val_1_list,'loss_val_1':loss_val_1_list}
results_df=pd.DataFrame(results)
results_df.to_csv(f'alex_val_2_results_{set}.csv',index=False)
os.chdir(cwd)



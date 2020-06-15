import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from alexnet_dann import alexnet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pandas as pd
import datetime
import os, sys
from plotting import plot
from matplotlib.pyplot import imshow, imsave

print(torch.__version__) ## Check pytorch version

# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

##HYPERPARAMETERS
SET = '42'        ##set of hyperparamaters
NUM_CLASSES = 7 # 7  
BATCH_SIZE = 256 # Higher batch sizes allows for larger learning rates changing the batch size, learning rate should change by the same factor 
LR = 1e-4           # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
NUM_EPOCHS = 30   # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20      # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down
LOG_FREQUENCY = 10
ALPHA = 0.05
# Define transforms for training phase
source_transform = transforms.Compose([transforms.Resize(256),      # Resizes short size of the PIL image to 256
                                      transforms.CenterCrop(224),  # Crops a central square patch of the image
                                                                   # 224 because torchvision's AlexNet needs a 224x224 input!
                                                                   # Remember this when applying different transformations, otherwise you get an error
                                      transforms.ToTensor(),       # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalizes tensor with mean and standard deviation
])
# Define transforms for the evaluation phase   
target_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))                                    
])
#Prepare Pytorch train/test Datasets
photo_dir='PACS/photo'
art_dir='PACS/art_painting'
cartoon_dir='PACS/cartoon'
sketch_dir='PACS/sketch'

photo_dataset = ImageFolder(photo_dir,transform = source_transform)
art_dataset = ImageFolder(art_dir,transform = target_transform)
sketch_dataset= ImageFolder(sketch_dir,transform = target_transform)
cartoon_dataset = ImageFolder(cartoon_dir, transform = target_transform)

source_dataset = photo_dataset
target_dataset = art_dataset
val_1_dataset = sketch_dataset
val_2_dataset = cartoon_dataset

# Check dataset sizes
print('Source Dataset: {}'.format(len(source_dataset)))
print('Target Dataset: {}'.format(len(target_dataset)))
print('Val Dataset 1: {}'.format(len(val_1_dataset)))
print('Val Dataset 2: {}'.format(len(val_2_dataset)))

# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
source_dataloader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
#val_1_dataloader = DataLoader(val_1_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
val_1_dataloader = DataLoader(val_2_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

iterations=max(len(val_1_dataloader),len(source_dataloader))
print(f'Iterations: {iterations}')

print(type(source_dataloader))

net=alexnet(pretrained=True)

net.classifier[6]=nn.Linear(4096,NUM_CLASSES) 
net.classifier_domain[6]=nn.Linear(4096,2)

##COPY WEIGHTS OF OTHER LAYERS
net.classifier_domain[1].weight.data=net.classifier[1].weight.data
net.classifier_domain[1].bias.data=net.classifier[1].bias.data
net.classifier_domain[4].weight.data=net.classifier[4].weight.data
net.classifier_domain[4].bias.data=net.classifier[4].bias.data

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

src_class_acc_list=[]
src_class_loss_list=[]
acc_target_list=[]
domain_loss_list=[]
domain_acc_list=[]
loss_target_list=[]
trg_domain_acc_list=[]
src_domain_acc_list=[]
val_1_domain_acc_list=[]
trg_class_accuracy_list=[]
val_1_class_accuracy_list=[]
val_1_class_loss_list=[]

# Start iterating over the epochs
for epoch in range(NUM_EPOCHS):
  print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, scheduler.get_lr()))
  src_dataloader_iterator = iter(source_dataloader)
  val_1_dataloader_iterator = iter(val_1_dataloader)

  running_loss=0
  src_class_running_loss=0
  src_class_running_corrects=0
  domain_corrects=0
  val_1_domain_running_corrects=0
  src_domain_running_corrects=0
  val_1_class_running_corrects=0
  val_1_class_running_loss=0

  for i in range(iterations):
    try:
      images , labels = next(src_dataloader_iterator)
      # Bring data over the device of choice
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      net.train()

      optimizer.zero_grad() #need to zeroing all gradients at each new batch pass

      src_class_outputs = net(images)
      # Compute loss based on output and ground truth
      src_class_loss = criterion(src_class_outputs,labels)

      _ , src_class_preds = torch.max(src_class_outputs.data, 1)

      src_class_running_corrects += torch.sum(src_class_preds == labels.data).data.item()
      
      src_domain_outputs = net(images, ALPHA)

      src_domain_labels=torch.zeros(BATCH_SIZE, dtype=torch.long).to(DEVICE)
      
      _, src_domain_preds = torch.max(src_domain_outputs.data,1)
      
      src_domain_running_corrects += torch.sum(src_domain_preds == src_domain_labels.data).data.item()
    
      # Compute loss based on output and ground truth
      src_domain_loss=criterion(src_domain_outputs,src_domain_labels)

      images, labels = next(val_1_dataloader_iterator)
      
      images=images.to(DEVICE)

      labels=labels.to(DEVICE)

      val_1_domain_outputs=net(images,ALPHA)
      
      val_1_domain_labels = torch.ones(BATCH_SIZE, dtype=torch.long).to(DEVICE)

      _, val_1_domain_preds=torch.max(val_1_domain_outputs.data,1)

      val_1_domain_running_corrects += torch.sum(val_1_domain_preds == val_1_domain_labels.data).data.item()
      # Compute loss based on output and ground truth
      val_1_domain_loss=criterion(val_1_domain_outputs,val_1_domain_labels)

      loss = src_class_loss + src_domain_loss + val_1_domain_loss
      running_loss += loss

      # Compute gradients for each layer and update weights
      loss.backward()  # backward pass: computes gradients

    except StopIteration:
      src_dataloader_iterator = iter(source_dataloader)
      images , labels = next(src_dataloader_iterator)
      # Bring data over the device of choice
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      net.train()

      optimizer.zero_grad() #need to zeroing all gradients at each new batch pass

      src_class_outputs = net(images)
      # Compute loss based on output and ground truth
      src_class_loss = criterion(src_class_outputs,labels)

      _ , src_class_preds = torch.max(src_class_outputs.data, 1)

      src_class_running_corrects += torch.sum(src_class_preds == labels.data).data.item()
      
      src_domain_outputs = net(images, ALPHA)

      src_domain_labels=torch.zeros(BATCH_SIZE, dtype=torch.long).to(DEVICE)
      
      _, src_domain_preds = torch.max(src_domain_outputs.data,1)
      
      src_domain_running_corrects += torch.sum(src_domain_preds == src_domain_labels.data).data.item()
    
      # Compute loss based on output and ground truth
      src_domain_loss=criterion(src_domain_outputs,src_domain_labels)

      images, labels = next(val_1_dataloader_iterator)
      
      images=images.to(DEVICE)

      labels=labels.to(DEVICE)

      val_1_domain_outputs=net(images,ALPHA)
      
      val_1_domain_labels=torch.ones(BATCH_SIZE, dtype=torch.long).to(DEVICE)

      _, val_1_domain_preds=torch.max(val_1_domain_outputs.data,1)

      val_1_domain_running_corrects += torch.sum(val_1_domain_preds == val_1_domain_labels.data).data.item()
      # Compute loss based on output and ground truth
      val_1_domain_loss=criterion(val_1_domain_outputs,val_1_domain_labels)

      loss = src_class_loss + src_domain_loss + val_1_domain_loss
      running_loss += loss

      # Compute gradients for each layer and update weights
      loss.backward()  # backward pass: computes gradients
    
    optimizer.step() # update weights based on accumulated gradients

    current_step += 1

    #VAL
    with torch.no_grad():
      net.train(False)
      labels=labels.to(DEVICE)
      val_1_class_outputs = net(images)
      _,val_1_class_preds = torch.max(val_1_class_outputs.data,1)
      val_1_class_running_corrects += torch.sum(val_1_class_preds == labels.data).data.item()
      val_1_class_loss = criterion(val_1_class_outputs , labels)
      val_1_class_running_loss += val_1_class_loss

  val_1_class_loss_avg = val_1_class_running_loss / iterations
  val_1_class_loss_list.append(val_1_class_loss_avg.data.item())
  val_1_class_accuracy = val_1_class_running_corrects / len(val_1_dataset)
  val_1_class_accuracy_list.append(val_1_class_accuracy)
  
  src_class_avg_loss = (src_class_loss / iterations).data.item()
  src_class_loss_list.append(src_class_avg_loss)

  domain_avg_loss = (src_domain_loss + val_1_domain_loss) / iterations
  domain_loss_list.append(domain_avg_loss.data.item())
  
  src_domain_accuracy = src_domain_running_corrects / float(iterations * BATCH_SIZE)
  src_domain_acc_list.append(src_domain_accuracy)
  
  val_1_domain_accuracy = val_1_domain_running_corrects / float((len(val_1_dataset)))
  val_1_domain_acc_list.append(val_1_domain_accuracy)
  
  domain_corrects = src_domain_running_corrects + val_1_domain_running_corrects
  domain_accuracy = domain_corrects / float(len(val_1_dataset) + (iterations * BATCH_SIZE))
  domain_acc_list.append(domain_accuracy)
  
  src_class_accuracy = src_class_running_corrects / float(iterations * BATCH_SIZE)
  src_class_acc_list.append(src_class_accuracy)
  #print(f'Epoch {epoch+1} class_loss: {class_avg_loss}')
  
  # Step the scheduler
  scheduler.step() 

plot(domain_loss_list,src_class_loss_list,'Loss','domain_loss','classification_loss','Losses vs epochs',f'dann_val_d+c_lossXepochs',SET)
plot(domain_acc_list,src_class_acc_list,'Accuracy','Domain_accuracy','Classification_accuracy','Accuracy vs Epochs',f'dann_val_d+c_accXepochs',SET)
plot(src_class_acc_list, val_1_class_accuracy_list,'Accuracy','Source_classification_accuracy','Validation_classification_accuracy','Accuracy vs Epochs',f'dann_val_accuraciesXepochs',SET)
plot(src_class_loss_list, val_1_class_loss_list,'Loss','Source_classification_loss','Salidation_classification_loss','Losses vs epochs',f'dann_val_lossesXepochs',SET)


cwd=os.getcwd()
os.chdir('files')
results={'domain_acc':domain_acc_list,'domain_loss':domain_loss_list,'class_acc_source':src_class_acc_list,'class_loss_source':src_class_loss_list,'class_acc_val_1':val_1_class_accuracy_list,'class_loss_val_1':val_1_class_loss_list}
results_df=pd.DataFrame(results)
results_df.to_csv(f'dann_validation_2_results_{SET}.csv',index=False)
os.chdir(cwd)
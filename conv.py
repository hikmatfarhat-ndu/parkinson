
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.classification
import torchvision as vision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader,Dataset, random_split
from torchmetrics.classification import BinaryConfusionMatrix
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
#torch.manual_seed(9)
transform = transforms.ToTensor()
dataset=ImageFolder("images",transform=transform)



class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # input is (*,3,512,512)
    self.model=nn.Sequential(
      nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
      
    # input is (*,32,510,510)
      nn.MaxPool2d(kernel_size=(2,2)),
# input is (*,32,255,255)
      nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
      nn.ReLU(),
    # input is (*,64,253,253)
      nn.MaxPool2d(kernel_size=(2,2)),
    # input is (*,64,126,126)
      nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
      nn.ReLU(),
    # input is (*,128,124,124)
      nn.MaxPool2d(kernel_size=(2,2)),
    # input is (*,128,62,62)
      nn.Flatten(),
    # input is (*,128x62x62)
      nn.Linear(in_features=62*62*128,out_features=1)

    )
   
  def forward(self,x):
    return self.model(x)
from torchmetrics.classification import BinaryAccuracy    
def get_accuracy(dataloader,model):
  accuracy=BinaryAccuracy().cuda()
#   total=len(dataloader)*dataloader.batch_size
#   correct=0
  for imgs,labels in dataloader:
    imgs,labels=imgs.cuda(),labels.cuda()
    outputs=model(imgs)
  # torchmetrics BinaryAccuracy automatically applies sigmoid
  # and threshold of (default) 0.5
    accuracy.update(outputs.squeeze(),labels)

  return accuracy.compute().item()

device='cuda' if torch.cuda.is_available() else 'cpu'

import torchmetrics
k=10
acc_v=np.empty(k)
auroc_v=np.empty(k)
precision_v=np.empty(k)
recall_v=np.empty(k)
roc_v=[]
conmat_v=[]
for v in range(k):
    accuracy=BinaryAccuracy().to(device)
    conmat=BinaryConfusionMatrix().to(device)
    precision=torchmetrics.classification.BinaryPrecision().to(device)
    recall=torchmetrics.classification.BinaryRecall().to(device=device)
    roc=torchmetrics.classification.BinaryROC().to(device=device)
    auroc=torchmetrics.classification.BinaryAUROC().to(device=device)
    print(f'Fold {v}')
    print('---------------------')
    print('---------------------')
    model=Net().to(device)
    optimizer=Adam(model.parameters())
    loss_fn=nn.BCEWithLogitsLoss()
    epochs=30
    dataset_train,dataset_test=random_split(dataset,lengths=[0.8,0.2])
    loader_train=DataLoader(dataset_train,batch_size=32,shuffle=True,num_workers=2)
    loader_test=DataLoader(dataset_test,batch_size=16,shuffle=False)

    for epoch in range(epochs):
      loop=tqdm(loader_train)
      loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
      epoch_loss=0.
      for (imgs,labels) in loop:
        optimizer.zero_grad()
        imgs=imgs.cuda()
        labels=labels.cuda()
        outputs=model(imgs)
        loss=loss_fn(outputs.squeeze(),labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss=0.9*epoch_loss+0.1*loss.item()
        loop.set_postfix(loss=epoch_loss)
    
      acc=get_accuracy(loader_test,model)
      print(acc)


    for data in loader_test:
      imgs,labels=data
      imgs=imgs.cuda()
      labels=labels.cuda()
      outputs=model(imgs).squeeze()

      accuracy.update(outputs,labels)
      recall.update(outputs,labels)
      roc.update(outputs,labels)
      auroc.update(outputs,labels)
      precision.update(outputs,labels)
      conmat.update(outputs,labels)

    acc_val=accuracy.compute()
    x=conmat.compute().cpu().numpy()
    precision_val=precision.compute()
    recall_val=recall.compute()
    auroc_val=auroc.compute()
    roc_val=roc.compute()
    a=[z.cpu().numpy() for z in roc_val[:2]]
    roc_v.append((roc_val[0].cpu().numpy(),roc_val[1].cpu().numpy()))
    acc_v[v]=acc_val.cpu().numpy()
    auroc_v[v]=auroc_val.cpu().numpy()
    precision_v[v]=precision_val
    recall_v[v]=recall_val
    conmat_v.append(x)
    plt.plot(a[0],a[1])
    plt.xlim(0,1.1)
    plt.ylim(0,1)
    plt.savefig(f'roc_{v}.png')
    #plt.clf()
    print(f'accuracy={acc_val} precision={precision_val}, recall={recall_val},auroc={auroc_val}')
    print(x)

print(f'accuracy_mean={acc_v.mean()}')
print(f'precision_mean={precision_v.mean()}')
print(f'recall_mean={recall_v.mean()}')
print(f'auroc_mean={auroc_v.mean()}')
tpr=roc_v[0][0]
fpr=roc_v[0][1]
for i in range(1,k):
  tpr+=roc_v[i][0]
  fpr+=roc_v[i][1]
tpr/=k
fpr/=k

#mat=np.empty((2,2))
mat=conmat_v[0]
for i in range(1,k):
  mat+=conmat_v[i]
mat=mat/k
plt.figure(figsize=(10,7))
sb.heatmap(x,xticklabels=['pos','neg'],yticklabels=['pos','neg'],annot=True,fmt=".0f")


# - The rows are the actual images and the columns are the prediction (How can you check?)
# - While the prediction accuracy is good albeit not impressive
# - From the confusion matrix we find justifications for the inaccuracies
# - For example
#     - most of the incorrect classifications of automobiles were classified as trucks
#     - most of the incorrect classifications of cats/dogs were classified as dogs/cats
#     





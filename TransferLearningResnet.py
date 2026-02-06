# 6 Epochs got accuracy of 82.58%s
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader 
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

learning_rate=.001
batch_size=100
output_size=10
epochs=6

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
    
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
    
])
cifar_data = datasets.CIFAR10(root='./data',train=True,download=True,transform=train_transform)
test_data=datasets.CIFAR10(root='./data',train=False,transform=test_transform,download=True)
train_size=len(cifar_data)
validation_size=int(0.2 * train_size)
train_size-=validation_size

train_data, val_data=random_split(cifar_data,[train_size,validation_size],)

train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(dataset=val_data,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)


model = models.resnet.resnet18(models.resnet.ResNet18_Weights.DEFAULT)
inFeatures=model.fc.in_features
model.fc=nn.Linear(inFeatures,10)

criterian=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)
for epoch in range(epochs):
    model.train()
    for index, (images, lables) in tqdm(enumerate(train_loader)):
        images=images.to(device)
        lables=lables.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss=criterian(outputs,lables)
        loss.backward()
        optimizer.step()
        if(index +1 ) % 100 ==0:
            print(f'Loss : {loss.item():.4f}')

    model.eval()
    val_loss=0
    for index, (images, lables) in tqdm(enumerate(val_loader)):
        images=images.to(device)
        lables=lables.to(device)
        val_loss+=criterian(outputs,lables).item()
    val_loss/=len(val_loader)
    scheduler.step(val_loss)
   
noOfsamples=0
noOfCorrect=0
with torch.no_grad():
     for i,(images,labels) in tqdm(enumerate(test_loader)):
        inputs=images.to(device)
        labels=labels.to(device)
        noOfsamples+= len(inputs)
        outputs=model(inputs)
        _,preds=torch.max(outputs,1)
        noOfCorrect+= torch.sum(preds==labels).item()

acc=noOfCorrect/noOfsamples*100
print(f'Accuracy of the model on test data:{acc} %')
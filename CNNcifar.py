# Accuracy of the model on test data:91.06 %. Initially it was 29% 
# after applying scheduler it reached to 78% 
# with data augmentation reached to 91.06%
# Around 12 hrs of model training on cpu.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

learning_rate=0.001
epochs=50
input_size=9216
hidden_size=700
output_size=10
batch_size=100



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
train_data=datasets.CIFAR10(root='./data',train=True,transform=train_transform,download=True)
test_data=datasets.CIFAR10(root='./data',train=False,transform=test_transform,download=True)

train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

class CNNCifar(nn.Module):
    def __init__(self, input_size, ouput_size):
        super(CNNCifar,self).__init__()
        self.conv2d1= nn.Conv2d(3,32,kernel_size=3,padding=1) # 32 * 32 * 32
        self.nb1= nn.BatchNorm2d(32)
        self.conv2d2=nn.Conv2d(32,64,kernel_size=3,padding=1) # 64 * 32 * 32
        self.nb2=nn.BatchNorm2d(64)
        self.pool1= nn.MaxPool2d(kernel_size=(2,2)) # 64 * 16 * 16
        self.drop1=nn.Dropout(p=.2)
        self.conv2d3= nn.Conv2d(64,128,kernel_size=3) # 128 * 14 * 14
        self.nb3= nn.BatchNorm2d(128)
        self.conv2d4=nn.Conv2d(128,256,kernel_size=3) # 256 * 12 * 12
        self.nb4=nn.BatchNorm2d(256)
        self.pool2= nn.MaxPool2d(kernel_size=(2,2)) # 256 * 6 * 6
        self.drop2=nn.Dropout(p=.2)
        self.l1 = nn.Linear(input_size,512) 
        self.l2 = nn.Linear(512,output_size)

    def forward(self, x):
        x=self.drop1(self.pool1(self.nb2(F.relu(self.conv2d2(self.nb1(F.relu(self.conv2d1(x))))))))
        x=self.drop2(self.pool2(self.nb4(F.relu(self.conv2d4(self.nb3(F.relu(self.conv2d3(x))))))))
        x = torch.flatten(x, 1)
        x=self.l2(F.relu(self.l1(x)))
        return x

model=CNNCifar(input_size,output_size)
from torchinfo import summary
summary(model=model,input_size=(1,3,32,32))

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=5e-4)
criterian=nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)

num_steps = len(train_loader)

for epoch in range(epochs):
    model.train()
    for i, (images,labels) in tqdm(enumerate(train_loader)):
        X = images.to(device)
        Y = labels.to(device)
        ypred=model(X)
        loss=criterian(ypred,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
             print(f'Epoch {epoch+1}/{epochs} step {i+1}/{num_steps} loss={loss.item():.4f}')

    model.eval()
    val_loss=0
    with torch.no_grad():
        for i,(images,labels) in tqdm(enumerate(test_loader)):
            inputs=images.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            val_loss+=criterian(outputs,labels).item()
        val_loss/=len(test_loader)
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

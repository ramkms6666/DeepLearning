# got the 98.33% accuracy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt 
import torch.nn.functional as F

# import datasets 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data=datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_data=datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())



epochs=4
input_size=784
hidden_size=650
output_size=10
batch_size=100
learning_rate=0.01
print(train_data.data.shape)
print(test_data.data.shape)

train_loder=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loder=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

class CnnNet(nn.Module):
    def __init__(self,input_size, output_size ):
        super(CnnNet,self).__init__()
        self.cnn1=nn.Conv2d(1,32,kernel_size=3) # b * 32 * 26 * 26
        self.pool1=nn.MaxPool2d(2,2) # b * 32 * 13 * 13
        self.cnn2=nn.Conv2d(32,64,kernel_size=3) # b * 64 * 11 * 11
        self.pool2=nn.MaxPool2d(2,2) # b * 64 * 5 * 5
        self.l1=nn.Linear(64*5*5,64) 
        self.l2=nn.Linear(64,10)

    def forward(self,x):
        x= self.pool1(F.relu(self.cnn1(x)))  # b * 32 * 13 * 13
        x= self.pool2(F.relu(self.cnn2(x))) # b * 64 * 5 * 5
        x=x.reshape(-1,64*5*5)
        x = self.l2(F.relu(self.l1(x)))
        return x 
            




model = CnnNet(input_size,output_size)  

from torchinfo import summary
summary(model=model,input_size=(1,28,28))


optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
criterian=nn.CrossEntropyLoss()


num_steps = len(train_loder)
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loder):
        X = images.to(device)
        Y = labels.to(device)
        ypred=model(X)
        loss=criterian(ypred,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
             print(f'Epoch {epoch+1}/{epochs} step {i+1}/{num_steps} loss={loss.item():.4f}')

    
# for epoch in epochs:
#     for item in enumerate(train_loder):
noOfsamples=0
noOfCorrect=0
with torch.no_grad():
     for i,(images,labels) in enumerate(test_loder):
        inputs=images.to(device)
        outputs=labels.to(device)
        noOfsamples+= len(inputs)
        outputs=model(inputs)
        _,preds=torch.max(outputs,1)
        noOfCorrect+= torch.sum(preds==labels).item()

acc=noOfCorrect/noOfsamples*100
print(f'Accuracy of the model on test data:{acc} %')

        
    



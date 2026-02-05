# got the 96.38% accuracy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt 

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

class NueralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NueralNet, self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.ReLU()
        self.l3=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        return self.l3(self.l2(self.l1(x)))

model = NueralNet(input_size,hidden_size,output_size)   
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
criterian=nn.CrossEntropyLoss()

def plotImages(images):
    print(images.shape)
    fig,axes= plt.subplots(5,5)
    for i,image in enumerate(images[:15]):
            print(image.shape)
            ax=axes[i // 3,i % 3] 
            ax.imshow(image[0])
            plt.axis('off')
    plt.tight_layout()  
    plt.show()
num_steps = len(train_loder)
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loder):
        X = images.reshape(-1,28*28).to(device)
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
        inputs=images.reshape(-1,28*28).to(device)
        outputs=labels.to(device)
        noOfsamples+= len(inputs)
        outputs=model(inputs)
        _,preds=torch.max(outputs,1)
        noOfCorrect+= torch.sum(preds==labels).item()

acc=noOfCorrect/noOfsamples*100
print(f'Accuracy of the model on test data:{acc} %')

        
    



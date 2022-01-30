import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split,DataLoader
import torch.nn as nn
import torch.nn.functional as F



dataset=MNIST(root='data/',download=True,transform=transforms.ToTensor())
train_ds,val_ds=random_split(dataset,[50000,10000])
batch_size=128
train_dl=DataLoader(train_ds,batch_size,shuffle=True)
val_dl=DataLoader(val_ds,batch_size)

input_size=28*28
num_classes=10

class Mnistmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(input_size,num_classes)
    def forward(self,x):
        x=x.reshape(-1,784)
        out=self.linear(x)
        return out
    def training_step(self,batch):
        images,labels= batch
        out=self(images)
        loss=F.cross_entropy(out,labels)
        return loss 
    def validation_step(self,batch):
        images,labels=batch
        out=self(images)
        loss=F.cross_entropy(out,labels)
        acc=accuracy(out,labels)
        return {'val_loss':loss,'val_acc':acc}  
    def validation_epoch_end(self,outputs):
        batch_losses=[x['val_loss'] for x in outputs] 
        epoch_loss=torch.stack(batch_losses).mean()
        batch_acc=[x['val_acc'] for x in outputs]
        epoch_acc=torch.stack(batch_acc).mean()
        return {'val_loss':epoch_loss.item(),'val_acc':epoch_acc.item()}
    def epoch_end(self,epoch,result):
        print("Epoch [{}] , val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,result['val_loss'],result['val_acc']))


def accuracy(outputs,labels):
    _,preds =torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

def fit(epochs,lr,model,train_dl,val_dl,opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(),lr)
    history=[]
    for epoch in range(epochs):
        for batch in train_dl:
            loss=model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result=evaluate(model,val_dl)  
        model.epoch_end(epoch,result)
        history.append(result)  
    return history    

def evaluate(model,val_dl):
    outputs=[model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

model=Mnistmodel()
model.load_state_dict(torch.load('test.pth'))
model.eval()
history1 = fit(10, 0.001, model, train_dl, val_dl)
torch.save(model.state_dict(), 'test.pth')

import torch
from torch.autograd import Variable

xdata=Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
ydata=Variable(torch.Tensor([[1.0],[2.0],[3.0]]))


class LRM(torch.nn.Module):
    def __init__(self) -> None:
        super(LRM,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        ypred=self.linear(x)
        return ypred
ourmodel=LRM()
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(ourmodel.parameters(),lr=0.01)  
for epoch in range(100):
    predy=ourmodel(xdata)
    loss=criterion(predy,ydata)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {},loss {}'.format(epoch,loss.item()))
newvar=Variable(torch.Tensor([[4.0]]))
predy=ourmodel(newvar)
print("predict (after training)",4,ourmodel(newvar).data[0][0])              
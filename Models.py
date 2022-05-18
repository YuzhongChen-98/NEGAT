import torch
import torch.nn as nn 
class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.conv1=nn.Conv1d(in_channels=3,out_channels=3,kernel_size=1,padding=0)
        self.conv2=nn.Conv1d(in_channels=116,out_channels=116,kernel_size=1,padding=0)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,Z,X):
        K=self.conv1(X.permute(0,2,1))# BS,x_c,x_dim
        Q=K.permute(0,2,1)# BS,x_dim,x_c
        V=self.conv2(Z.permute(0,2,1))# Bs,z_c,z_dim
        attention=self.softmax(torch.matmul(Q,K))#BS,x_dim,x_dim
        out=torch.bmm(attention,V).permute(0,2,1)#BS,z_dim,z_c
        return out

class NEGAN(nn.Module):
    def __init__(self,layer):
        super(NEGAN,self).__init__()
        self.layer =layer
        self.relu  =nn.ReLU()
        self.atten =nn.ModuleList([Attention() for i in range(layer)])
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ParameterList([nn.Parameter(torch.randn((3,3),dtype=torch.float32)) for i in range(layer)])
        self.edge_w=nn.ParameterList([nn.Parameter(torch.randn((116,116),dtype=torch.float32)) for i in range(layer)])
        self.line_n=nn.ModuleList([nn.Sequential(nn.Linear(116*3,128),nn.ReLU(),nn.BatchNorm1d(128)) for i in range(layer+1)])
        self.line_e=nn.ModuleList([nn.Sequential(nn.Linear(116*116,128*3),nn.ReLU(),nn.BatchNorm1d(128*3)) for i in range(layer+1)])
        self.clase =nn.Sequential(nn.Linear(128*4*(self.layer+1),1024),nn.Dropout(0.2),nn.ReLU(),
                                   nn.Linear(1024,2))
        self.ones=nn.Parameter(torch.ones((116),dtype=torch.float32),requires_grad=False)
        self._initialize_weights()
    # params initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def normalized(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        A=A+torch.diag(self.ones)
        d=A.sum(1)
        D=torch.diag(torch.pow(d,-1))
        A=D.mm(A).reshape(1,116,116)
        for i in range(1,n):
            A1=Z[i,:,:]+torch.diag(self.ones)
            d=A1.sum(1)
            D=torch.diag(torch.pow(d,-1))
            A1=D.mm(A1).reshape(1,116,116)
            A=torch.cat((A,A1),0)
        return A
        
    def update_A(self,Z):
        n=Z.size()[0]
        A=Z[0,:,:]
        Value,_=torch.topk(torch.abs(A.view(-1)),int(116*116*args.thed))
        A=(torch.abs(A)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
        A=A.reshape(1,116,116)
        for i in range(1,n):
            A2=Z[i,:,:]
            Value,_=torch.topk(torch.abs(A2.view(-1)),int(116*116*args.thed))
            A2=(torch.abs(A2)>=Value[-1])+torch.tensor(0,dtype=torch.float32)
            A2=A2.reshape(1,116,116)
            A=torch.cat((A,A2),0)
        return A
        
    def forward(self,X,Z):
        n=X.size()[0]
        XX=self.line_n[0](X.view(n,-1))
        ZZ=self.line_e[0](Z.view(n,-1))
        for i in range(self.layer):
            A=self.atten[i](Z,X)
            Z1=torch.matmul(A,Z)
            Z2=torch.matmul(Z1,self.edge_w[i])
            Z=self.relu(self.norm_e[i](Z2))+Z
            ZZ=torch.cat((ZZ,self.line_e[i+1](Z.view(n,-1))),dim=1)
            X1=torch.matmul(A,X)
            X1=torch.matmul(X1,self.node_w[i])
            X=self.relu(self.norm_n[i](X1))+X
            XX=torch.cat((XX,self.line_n[i+1](X.view(n,-1))),dim=1)
        XZ=torch.cat((XX,ZZ),1)
        y=self.clase(XZ)
        return y
from dataloader import *
from PointTransformerBlock import *
from TransitionDownBlock import *

class Whole(nn.Module):
    def __init__(self):
        super(Whole, self).__init__()
        self.mlp1 = nn.Linear(3, 32)
        self.PTB1 = PointTransformerBlock(32)
        self.TDB1 = TransitionDown(512,16,32,64)
        self.PTB2 = PointTransformerBlock(64)
        self.TDB2 = TransitionDown(128,16,64,128)
        self.PTB3 = PointTransformerBlock(128)
        self.TDB3 = TransitionDown(32,16,128,256)
        self.PTB4 = PointTransformerBlock(256)
        self.TDB4 = TransitionDown(8,16,256,512)
        self.maxpool = nn.MaxPool1d(kernel_size=8)
        self.mlp2 = nn.Linear(512, 40)

    def forward(self,x):
        posi = x
        feature = self.mlp1(x)
        feature, posi = self.PTB1(feature, posi)
        feature, posi = self.TDB1(feature, posi)
        feature, posi = self.PTB2(feature, posi)
        feature, posi = self.TDB2(feature, posi)
        feature, posi = self.PTB3(feature, posi)
        feature, posi = self.TDB3(feature, posi)
        feature, posi = self.PTB4(feature, posi)
        feature, posi = self.TDB4(feature, posi)
        feature = feature.permute(0,2,1) #torch.Size([12, 512, 8])
        feature = self.maxpool(feature).squeeze() #torch.Size([12, 512])
        feature = (self.mlp2(feature))

        return feature, posi

# if __name__ == '__main__':
#     data = ModelNet40(split='test',path='modelnet40_normal_resampled/')
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
#     model = Whole()
#     for point,label in DataLoader:
#         print(point.shape)  #[b,n,3] torch.Size([12, 2048, 3])
#         print(label.shape)  #[b,1]  torch.Size([12, 1])
#         feature, posi = model(point)
#         print(feature.shape)  #torch.Size([12, 2048, 32])
#         print(posi.shape) #torch.Size([12, 2048, 3])
#         print('--------')

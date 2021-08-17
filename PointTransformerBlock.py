import torch.nn as nn
import torch
from torch import einsum
from dataloader import *

# def knn(point,feature_set,position_set,k=16): #find the nearest k points to point from dataset
#     #input[3] input[2048,32] input[2048,3]
#     #output[k,32] output[k,3]
#     difference = position_set - point.reshape(1,3).repeat(position_set.shape[0],1)
#     distance = torch.sqrt(torch.sum(difference ** 2, dim=1))
#     con = torch.cat((position_set, distance.reshape(position_set.shape[0],1)), dim=1) #[2048,4]
#     _, indices = torch.sort(con, dim=0)
#     index = indices[:,3]
#     con = con[index]
#     k_points = con[1:k+1,:3]
#
#     feature_set = torch.cat((feature_set, distance.reshape(position_set.shape[0], 1)), dim=1)
#     feature_set = feature_set[index]
#     k_feature = feature_set[1:k+1,:-1]
#
#     return k_feature,k_points


# def create_k_nearst_matrix(x,p): #torch.Size([12, 2048, 32])  #torch.Size([12, 2048, 3])
#     x_i = []
#     p_i = []
#     for i in range(x.shape[0]):
#         x_j = []
#         p_j = []
#         for j in range(x.shape[1]):
#             a, b = knn(p[i][j], x[i], p[i])
#             x_j.append(a[None,])
#             p_j.append(b[None,])
#         x_j = torch.cat(x_j, dim=0)  # torch.Size([2048, 16, 32])
#         p_j = torch.cat(p_j, dim=0)  # torch.Size([2048, 16, 3])
#         x_i.append(x_j[None,])
#         p_i.append(p_j[None,])
#     x_i = torch.cat(x_i, dim=0)  # torch.Size([12, 2048, 16, 32])
#     p_i = torch.cat(p_i, dim=0)  # torch.Size([12, 2048, 16, 3])
#     return x_i, p_i


class PointTransformer(nn.Module):
    def __init__(self, input_dim=32, pos_inner_dim =64, value_inner_dim = 4, k=16):  #PTbolck中间维度,输入维度
        super(PointTransformer, self).__init__()
        dim = input_dim
        self.k = 16
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)
        self.mlp_posi = nn.Sequential(
            nn.Linear(3, pos_inner_dim),
            nn.ReLU(),
            nn.Linear(pos_inner_dim, dim)
        )
        self.mlp_value = nn.Sequential(
            nn.Linear(dim, value_inner_dim),
            nn.ReLU(),
            nn.Linear(value_inner_dim, dim)
        )
    def forward(self, x,p): #[bs:12,N:2048,feature:32] #[bs:12,N:2048,position:3]
        #x_k, p_k = create_k_nearst_matrix(x, p)# torch.Size([12, 2048, 16, 32]) torch.Size([12, 2048, 16, 3])
        #中求N个点的k个最近的点的索引
        distance = torch.sum((p[:, :, None, :] - p[:, None, :, :]) ** 2, -1)  # torch.Size([12, 2048, 2048])
        index = distance.argsort()[..., 1:self.k + 1]  # torch.Size([12, 2048, k]) k=16
        #posi_diff = p[:,:,None,:] - p_k #torch.Size([12, 2048, 16, 3])
        #根据索引得到值
        idx = index.reshape(index.shape[0], -1) # torch.Size([12, 2048*k]) k=16
        new_p = torch.gather(p, 1, idx[..., None].expand(-1, -1, p.size(-1)))
        new_p =  new_p.reshape(*index.shape, -1) #torch.Size([12, 2048, 16, 3])
        #position差过网络
        posi_diff = p[:, :, None] - new_p
        posi_diff = self.mlp_posi(posi_diff) #torch.Size([12, 2048, 16, 32])

        x_i = self.A(x)
        x_j1 = self.B(x)
        x_j2 = self.C(x)

        # 根据索引得到值
        idx_1 = index.reshape(index.shape[0], -1)  # torch.Size([12, 2048*k]) k=16
        x_j1 = torch.gather(x_j1, 1, idx_1[..., None].expand(-1, -1, x.size(-1)))
        x_j1 = x_j1.reshape(*index.shape, -1)  # torch.Size([12, 2048, 16, 32])
        # 根据索引得到值
        idx_2 = index.reshape(index.shape[0], -1)  # torch.Size([12, 2048*k]) k=16
        x_j2 = torch.gather(x_j2, 1, idx_2[..., None].expand(-1, -1, x.size(-1)))
        x_j2 = x_j2.reshape(*index.shape, -1)  # torch.Size([12, 2048, 16, 32])
        # print(x_i.shape)
        # print(x_j1.shape)
        # print(posi_diff.shape)
        x = x_i[:,:,None,:] - x_j1 + posi_diff
        x = self.mlp_value(x)
        x = x.softmax(dim=-2)   # torch.Size([12, 2048, 16, 32])

        posi = posi_diff + x_j2 #torch.Size([12, 2048, 16, 32])
        y = einsum('b i j d, b i j d -> b i d', x, posi)  #torch.Size([12, 2048, 32])
        return y,p


class PointTransformerBlock(nn.Module):
    def __init__(self,d_input_layer,d_inner_layer=32):
        super(PointTransformerBlock, self).__init__()
        self.dim_inner = d_inner_layer
        self.dim_input = d_input_layer
        self.linear1 = nn.Linear(self.dim_input, self.dim_inner)
        self.linear2 = nn.Linear(self.dim_inner, self.dim_input)
        self.PTT = PointTransformer(input_dim=32)

    def forward(self,x,p): #[bs:12,N:2048,feature:32/64/128/256]
        y = self.linear1(x)
        y,p = self.PTT(y,p)
        y = self.linear2(y)
        y = y + x
        return y,p


# if __name__ == '__main__':
#     data = ModelNet40(split='test',path='modelnet40_normal_resampled/')
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
#     model = PointTransformerBlock(d_input_layer=32)
#     x = torch.randn(12, 2048, 32)
#     for point,label in DataLoader:
#         print(point.shape)  #[b,n,3] torch.Size([12, 2048, 3])
#         print(label.shape)  #[b,1]  torch.Size([12, 1])
#         feature, posi = model(x,point)
#         print(feature.shape)  #torch.Size([12, 2048, 32])
#         print(posi.shape) #torch.Size([12, 2048, 3])
#         print('--------')






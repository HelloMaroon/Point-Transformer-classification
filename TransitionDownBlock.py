import torch.nn as nn
import torch

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B,N,3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    batchsize, ndataset, dimension = xyz.shape
    #to方法Tensors和Modules可用于容易地将对象移动到不同的设备（代替以前的cpu()或cuda()方法）
    # 如果他们已经在目标设备上则不会执行复制操作
    centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batchsize, ndataset).to(device) * 1e10
    #randint(low, high, size, dtype)
    # torch.randint(3, 5, (3,))->tensor([4, 3, 4])
    farthest =  torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
    #batch_indices=[0,1,...,batchsize-1]
    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
    for i in range(npoint):
        # 更新第i个最远点
        centroids[:,i] = farthest
        # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
        # 计算点集中的所有点到这个最远点的欧式距离
        #等价于torch.sum((xyz - centroid) ** 2, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        #取出每一行的最大值构成列向量，等价于torch.max(x,2)
        farthest = torch.max(distance, -1)[1]
    return centroids  #[B, npoint] 记录保留下来的点的索引

def kNN(x,p,npoint,k=16):
    #x = torch.randn(12, N, 32)
    #p = torch.randn(12, N, 3)
    #new_x = torch.Size([12, npoint, k, 64])
    #new_p = torch.Size([12, npoint, 3])
    #简化点云，只取npoint个点
    index = farthest_point_sample(p, npoint)  #torch.Size([12, npoint])
    new_x = torch.gather(x,1,index[...,None].expand(-1,-1,x.shape[-1])) #torch.Size([12, npoint, 32])
    new_p = torch.gather(p,1,index[...,None].expand(-1,-1,p.shape[-1])) #torch.Size([12, npoint, 3])
    #在总共N各点中求npoint个点的k个最近的点的索引
    distance = torch.sum((new_p[:,:,None,:] - p[:,None,:,:]) ** 2, -1) #torch.Size([12, npoint, 2048])
    knn_index = distance.argsort()[...,1:k+1]  #torch.Size([12, npoint, k]) k=16
    #得到npoint个点的最近的k个点的特征
    knn_index_new = knn_index.reshape(knn_index.shape[0],-1) #torch.Size([12, npoint* k])
    feature = torch.gather(x,1,knn_index_new[...,None].expand(-1,-1,x.shape[-1])) #torch.Size([12, npoint* k,32])
    feature = feature.reshape(*knn_index.shape,-1) #torch.Size([12, npoint, k, 32])
    #最终特征两部分，一部分是做差，一部分是new_x中复制得到的表示此点的特征
    new_x = new_x[:, :, None, :] - feature #torch.Size([12, npoint, k, 32])
    # new_x = torch.cat([feature_diff, new_x[:, :, None, :].repeat(1, 1, k, 1)], dim=-1) #torch.Size([12, npoint, k, 64])

    return new_x, new_p #new_x = torch.Size([12, npoint, k, 64])#new_p = torch.Size([12, npoint, 3])


class TransitionDown(nn.Module):
    def __init__(self,npoint=16,k=16,input_feature=32,ouput_feature=64):
        super(TransitionDown, self).__init__()
        self.npoint = npoint
        self.k = k
        self.input_feature = input_feature
        self.ouput_feature = ouput_feature
        self.mlp = nn.Sequential(
            nn.Conv1d(self.input_feature, self.ouput_feature, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ouput_feature),
            nn.ReLU(True),
        )

    def forward(self, x, p):
        new_x, new_p = kNN(x,p,self.npoint,self.k)
        B, npoint, nsample,dim = new_x.shape
        # new_p: sampled points position data, [B, npoint, 3]
        # new_x: sampled points data, [B, npoint, k, 64]
        #new_x = new_x.permute(0,3,2,1)  #[B, 64, k, npoint]
        new_x = new_x.reshape(B,-1,dim)#[B, npoint*k, 32]
        new_x = new_x.permute(0, 2, 1) #[B, 32, npoint*k]
        new_x = self.mlp(new_x) #[B, 64, npoint*k]
        new_x = new_x.reshape(B,self.ouput_feature, npoint,-1).permute(0, 2, 3,1)#(B, M, k, out_channels)

        new_x = torch.max(new_x, dim=2)[0]  # y: (B, M, out_channels)
        return new_x, new_p
import numpy as np
import os
from torch.utils.data import Dataset


DATA_PATH='./modelnet40_normal_resampled/'
class ModelNet40(Dataset):
    def __init__(self,split='train',path=DATA_PATH,points_num=2048):
        self.root = path
        self.points_num = points_num

        #['airplane',...] 40
        self.classes=[line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_shape_names.txt'))]
        #['airplane_0001,...] 物体数量 9843/2468
        pt_names = {}
        pt_names['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]  # list
        pt_names['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        #['airplane',...] 物体数量
        self.labels = ['_'.join(x.split('_')[:-1]) for x in pt_names[split]]
        #['./modelnet40_normal_resampled/airplane/airplane_0001.txt' 物体数量
        self.datapath = [os.path.join(self.root,self.labels[i],pt_names[split][i]+'.txt') for i in range(len(self.labels))]

        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __getitem__(self, index):
        label = self.classes.index(self.labels[index])
        label = np.array([label])
        point_cloud = np.loadtxt(self.datapath[index], delimiter=',').astype(np.float32)
        point_cloud = point_cloud[:self.points_num,:3]   #取2048,可改进
        #归一化
        center = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - center
        radius = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1))) #最大半径
        point_cloud = point_cloud / radius

        return point_cloud, label

    def __len__(self):
        return len(self.datapath)



# if __name__ == '__main__':
#     data = ModelNet40(split='test',path='modelnet40_normal_resampled/')
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
#     for point,label in DataLoader:
#         print(point.shape)  #[b,n,3] torch.Size([12, 2048, 3])
#         print(label.shape)  #[b,1]  torch.Size([12, 1])

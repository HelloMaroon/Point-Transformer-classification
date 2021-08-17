from model import *
from tqdm import tqdm
import time
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Whole()
# model.train(True)
model = model.to(device)
# model = nn.DataParallel(model)
# model.train(True)

#cross entropy loss for classification
ce_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

PATH = './modelnet40_normal_resampled/'
train_data = ModelNet40(split='train',path=PATH)
train_set = torch.utils.data.DataLoader(train_data, batch_size=12, shuffle=True)

test_data = ModelNet40(split='test',path=PATH)
test_set = torch.utils.data.DataLoader(test_data, batch_size=12, shuffle=True)


start_time = time.time()
for epoch in range(50):
    model.train()
    total_number = 0
    corrects = 0
    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_set)
    for point,label in train_bar:
        label = label.type(torch.LongTensor)
        label = label.squeeze(1)
        point = point.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        feature, posi = model(point)

        #print(feature.dtype)torch.float32
        #print(label.dtype)torch.int64
        # cross-entropy loss for all the three modalities
        loss = ce_criterion(feature, label)
        loss.backward()
        optimizer.step()

        # total loss
        total_loss += loss.item()
        total_num += point.size(0)

        _, pred = feature.topk(1, 1, True)
        correct = pred.T.eq(label.view(1, -1))
        #print(correct.dtype)#torch.bool
        corrects += correct.float().sum().item()
        # print(correct_img,correct_mesh,correct_pt,total_num)
        # print(correct_imgs,correct_meshes,correct_pts,total_num)
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f} Acc:{:.2f}%'.format(epoch+1, 50, total_loss / total_num, torch.true_divide(corrects, total_num) * 100))

    model.eval()

    corrects, total_num, test_bar = 0, 0, tqdm(test_set)
    with torch.no_grad():

      for point, label in test_bar:
        label = label.type(torch.LongTensor)
        label = label.squeeze(1)
        point = point.to(device)
        label = label.to(device)

        feature, posi = model(point)

        total_num += point.size(0)
        _, pred = feature.topk(1, 1, True)
        correct = pred.T.eq(label.view(1, -1))
        corrects += correct.float().sum().item()
        test_bar.set_description(
            'Test Epoch: [{}/{}] Accuracy: {:.4f}'.format(epoch+1, 50, torch.true_divide(corrects, total_num) * 100))

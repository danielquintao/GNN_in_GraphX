import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from dgl.nn import SAGEConv

from dgl.data import PPIDataset
dataset = dgl.data.PPIDataset()

class GNN(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GNN, self).__init__()
        self.conv = SAGEConv(in_feats, num_classes, "mean")

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        h = F.sigmoid(h)
        return h

print("default device (seems to be CPU in DGL):")

def train(graph_list, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=20.)

    for e in range(10):
        epoch_time = 0
        epoch_loss = 0
        epoch_acc = 0
        for g in graph_list:
            features = g.ndata['feat']
            labels = g.ndata['label']
            # Time
            start = time()
            # Forward
            logits = model(g, features)
            # Compute prediction
            pred = (logits > 0.5) * 1.0
            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.binary_cross_entropy(logits, labels)
            epoch_loss += loss/20
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Time
            epoch_time += time() - start
            # Compute accuracy on training/validation/test
            train_acc = (pred == labels).float().mean()
            epoch_acc += train_acc/20
        print(epoch_time, epoch_loss, epoch_acc)
model = GNN(50, 121)
train(dataset, model)
print('is in CUDA?', next(model.parameters()).is_cuda)

# print('GPU:')
# dataset_cuda = [g.to('cuda') for g in dataset]
# model_cuda = GNN(50, 121).to('cuda')
# train(dataset_cuda, model_cuda)
# print('is in CUDA?', next(model_cuda.parameters()).is_cuda)



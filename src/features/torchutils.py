
import h5py
import torch, math
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ListFolder(Dataset):
    def __init__(self, root, image_paths, labels_path, routing_idx_path=None, mask_values_path=None, isACE=False,  offset=0, is_train=False, transform=None):
        super(ListFolder, self).__init__()
        # Get image list and labels per index
        self.root = root
        self.offset = offset
        self.image_paths = h5py.File(image_paths, "r")["store_list"]
        self.labels = h5py.File(labels_path, "r")["store_list"]
        self.is_train = is_train
        self.isACE = isACE
      
        if self.is_train: 
            self.routing_idx = h5py.File(routing_idx_path, "r")["store_list"]
            if self.isACE:
                self.mask_values = h5py.File(mask_values_path, "r")["store_list"]
            
        self.transform = transform
        assert(len(self.image_paths)==len(self.labels))

    def __getitem__(self, index):
        # get the index from the routing index  
        sample_mask_value = np.array([1.0])
        if self.is_train: 
            sel_idx = self.routing_idx[index+self.offset]
            if self.isACE:
                sample_mask_value = self.mask_values[index+self.offset]
        else:
            sel_idx = index+self.offset
        
        # get the corresponding image and label
        img_path = self.root+'/'+self.image_paths[sel_idx].decode("utf-8").strip()
        label = self.labels[sel_idx]
        sample = pil_loader(img_path)
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, sample_mask_value, sel_idx, index

    def __len__(self):
        assert(len(self.image_paths)==len(self.labels)), 'Length of image path array and labels different'
        if self.is_train:
             return len(self.routing_idx)-self.offset
        else:
             return len(self.labels) -self.offset


class LinearLR(_LRScheduler):
    r"""Set the learning rate of each parameter group with a linear
    schedule: :math:`\eta_{t} = \eta_0*(1 - t/T)`, where :math:`\eta_0` is the
    initial lr, :math:`t` is the current epoch or iteration (zero-based) and
    :math:`T` is the total training epochs or iterations. It is recommended to
    use the iteration based calculation if the total number of epochs is small.
    When last_epoch=-1, sets initial lr as lr.
    It is studied in
    `Budgeted Training: Rethinking Deep Neural Network Training Under Resource
     Constraints`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Total number of training epochs or iterations.
        last_epoch (int): The index of last epoch or iteration. Default: -1.
        
    .. _Budgeted Training\: Rethinking Deep Neural Network Training Under
    Resource Constraints:
        https://arxiv.org/abs/1905.04753
    """

    def __init__(self, optimizer, T, last_epoch=-1):
        self.T = float(T)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        rate = 1 - self.last_epoch/self.T
        return [rate*base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        
        return out 


class EmbedLinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, embed_size, num_classes, cosfc=False):
        super(EmbedLinearClassifier, self).__init__()
        self.embed = None
        self.embed = nn.Linear(dim, embed_size)
        self.norm = nn.Sequential(nn.BatchNorm1d(embed_size), nn.ReLU(inplace=True))
        self.embed.weight.data.normal_(mean=0.0, std=0.01)
        self.embed.bias.data.zero_()
        if cosfc:
            self.fc = CosineLinear(embed_size, num_classes)
        else:
            self.fc = nn.Linear(embed_size, num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        if self.embed is not None:
            x = self.embed(x)
            x = self.norm(x)
        return self.fc(x)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

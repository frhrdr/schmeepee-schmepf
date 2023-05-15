# code taken from https://github.com/nanekja/JovianML-Project

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from collections import OrderedDict

import numpy as np
import torch.nn as nn
# import torch.nn.functional as nnf
import sys
# sys.path.append('/home/mijungp/dp-gfmn/code/')
sys.path.append('/mnt/d/dp-gfmn/code/')
# from util import get_image_size_n_feats, LOG
from collections import OrderedDict
import colorlog
LOG = colorlog.getLogger(__name__)

def get_image_size_n_feats(net, verbose=False, image_size=32, use_cuda=False):
  """
  return two list:
  - list of size of output (image) for each layer
  - list of size of total number of features (nFeatMaps*featmaps_height,featmaps_width)
  """
  if use_cuda:
    _, layers = net(torch.randn(1, 3, image_size, image_size).cuda())
  else:
    _, layers = net(torch.randn(1, 3, image_size, image_size))

  layer_img_size = []
  layer_num_feats = []
  for layer in reversed(layers):
    if len(layer.size()) == 4:
      layer_img_size.append(layer.size(2))
      layer_num_feats.append(layer.size(1)*layer.size(2)*layer.size(3))
    elif len(layer.size()) == 2:
      layer_img_size.append(1)
      layer_num_feats.append(layer.size(1))
    else:
      assert 0, f'not sure how to handle this layer size {layer.size()}'
  if verbose:
    LOG.info("# Layer img sizes: {}".format(layer_img_size))
    LOG.info("# Layer num feats: {}".format(layer_num_feats))

  return layer_img_size, layer_num_feats

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out, out2 = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out, out2 = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))



class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
    self.relu1 = nn.ReLU(True)
    self.relu2 = nn.ReLU(True)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes, track_running_stats=True)
      )

  def forward(self, x):
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = self.relu2(out)
    return out


class ResNet(ImageClassificationBase):
  def __init__(self, block, num_blocks, num_classes=10, image_size=32, get_perceptual_feats=False):
    super(ResNet, self).__init__()
    self.in_planes = 64

    # assert image_size % 32 == 0, f'image size {image_size} not supported'
    # n_max_pools = image_size // 32
    n_max_pools = 1

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512*block.expansion*n_max_pools*n_max_pools, num_classes)

    self._initialize_weights()
    self.get_perceptual_feats = get_perceptual_feats
    self.layer_feats = OrderedDict()

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    feature_list = []
    x = F.relu(self.bn1(self.conv1(x)))
    # first layer was not hooked, therefore we have to add its result manually
    feature_list.append(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = F.avg_pool2d(x, 4)
    x = x.view(x.size(0), -1)
    out = self.linear(x)
    if self.get_perceptual_feats:
      for k, v in self.layer_feats.items():
        # LOG.warning(f'{k}: {v.shape}')  # first time, layer feats is empty, then adds all layers
        feature_list.append(v)
      feature_list.append(out)
      return out, feature_list
    else:
      return out

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _get_hook(self, layer_num, layer):
    def myhook(_module, _input, _out):
      self.layer_feats[layer_num] = _out
    layer.register_forward_hook(myhook)


def resnet_18(get_perceptual_feats=False, num_classes=10, image_size=32):
  net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, image_size=image_size,
               get_perceptual_feats=get_perceptual_feats)
  if get_perceptual_feats:
    img_size_l, n_feats_l = get_image_size_n_feats(net, image_size=image_size)
    net.image_size_per_layer = np.array(img_size_l)
    net.n_features_per_layer = np.array(n_feats_l)

    # registers a hook for each RELU layer
    layer_num = 0
    for feature_layers in [net.layer1, net.layer2, net.layer3, net.layer4]:
      for res_block in feature_layers:
        for modl in res_block.modules():
          if str(modl)[0:4] == 'ReLU':
            LOG.debug("# registering hook module {} ".format(str(modl)))
            net._get_hook(layer_num, modl)
            layer_num += 1

  img_size_l, n_feats_l = get_image_size_n_feats(net, image_size=image_size)
  net.image_size_per_layer = np.array(img_size_l)
  net.n_features_per_layer = np.array(n_feats_l)
  return net



# class ResNet(ImageClassificationBase):
#     def __init__(self, in_channels=3, num_classes=10, get_perceptual_feats=True):
#         super().__init__()
#
#         self.conv1 = conv_block(in_channels, 64)
#         self.conv2 = conv_block(64, 128, pool=True)
#         self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
#
#         self.conv3 = conv_block(128, 256, pool=True)
#         self.conv4 = conv_block(256, 512, pool=True)
#         self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
#
#         # self.classifier = nn.Sequential(nn.MaxPool2d(3),
#         #                                 nn.Flatten(),
#         #                                 nn.Linear(512, num_classes))
#
#         self.maxpool = nn.MaxPool2d(3)
#         self.flatten = nn.Flatten()
#         self.l1 = nn.Linear(512, num_classes)
#
#         self.get_perceptual_feats = get_perceptual_feats
#         self.layer_feats = OrderedDict()
#
#     def forward(self, xb):
#
#         out = self.conv1(xb)
#         out = self.conv2(out)
#         out = self.res1(out) + out
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.res2(out) + out
#
#         output = self.maxpool(out)
#         flattened_output = self.flatten(output)
#         out = self.l1(flattened_output)
#
#         # out = self.classifier(out)
#         # return output, flattened_output
#         feature_list = []
#         if self.get_perceptual_feats:
#           for k, v in self.layer_feats.items():
#             # LOG.warning(f'{k}: {v.shape}')  # first time, layer feats is empty, then adds all layers
#             feature_list.append(v)
#           feature_list.append(out)
#           return out, feature_list
#         else:
#           return out
#
#
#
# ############################ classifier using VGG features #################################
#
# class Classifier_ResNet(nn.Module):
#
#     def __init__(self, features):
#         super(Classifier_ResNet, self).__init__()
#
#         self.features = features
#         self.parameter = Parameter(torch.zeros(512,10), requires_grad=True)
#         # self.l1 = nn.Linear(512,10)
#
#     def forward(self, x):  # x is mini_batch_size by input_dim
#
#         _, x = self.features(x)
#         # output = self.l1(x)
#         output = torch.matmul(x, self.parameter)
#
#         return output



import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

### Class used for computing the loss entropy
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        return entropy_loss(x)
        
def entropy_loss(logits):
    p_softmax = F.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to, used for numerical stability
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))
    
### Function to unnormalize photos to visualize them in a proper way    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

### Function to visualize a grid of RGB and depth images
def print_images(S_dataloader):
  train_iter = iter(S_dataloader)
  im_RGB, im_depth, labels = train_iter.next()
  unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  for im1 in im_RGB:
    im1 = unorm(im1)
    im1 = np.uint8(im1) * 255
  for im2 in im_depth:
    im2 = unorm(im2)
    im2 = np.uint8(im2) * 255
  grid = torchvision.utils.make_grid(im_RGB) 
  plt.figure(figsize=(12, 48))
  plt.imshow(grid.numpy().transpose((1, 2, 0)))
  plt.axis('off')
  grid = torchvision.utils.make_grid(im_depth)
  plt.figure(figsize=(12, 48))
  plt.imshow(grid.numpy().transpose((1, 2, 0)))
  plt.axis('off')
  
### Function to visualize the first two RGB and depth images from a dataloader and the relative absolute or relative rotation angle
def print_images_labels(dataloader):
  train_iter = iter(dataloader)
  im_RGB, im_depth, cos, sin, labels = train_iter.next()
  angle = (np.arctan2(sin[0], cos[0])*180)/(np.pi)
  unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  for im1 in im_RGB:
    im1 = unorm(im1)
    im1 = np.uint8(im1) * 255
  for im2 in im_depth:
    im2 = unorm(im2)
    im2 = np.uint8(im2) * 255

  grid_rgb = torchvision.utils.make_grid(im_RGB[0]) 
  plt.figure(figsize=(3, 3))
  plt.imshow(grid_rgb.numpy().transpose((1, 2, 0)))
  plt.axis('off')

  grid_depth = torchvision.utils.make_grid(im_depth[0]) 
  plt.figure(figsize=(3, 3))
  plt.imshow(grid_depth.numpy().transpose((1, 2, 0)))
  plt.axis('off')
  print(cos[0])
  print(sin[0])
  print("Angle {} labels {}".format(angle, labels[0]))
  
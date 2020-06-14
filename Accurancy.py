from tqdm.notebook import tqdm
import torch

### Test accuracy when using both modalities, rgb and depth
def test_acc_rgbd(net, test, DEVICE):
  net.train(False) 
  running_corrects_label = 0
  contatore = 0

  for rgb, depth, labels in tqdm(test):
    rgb = rgb.to(DEVICE)
    depth = depth.to(DEVICE)
    labels = labels.to(DEVICE)

    lab = net(rgb, depth)
    
    _, preds = torch.max(lab.data, 1)

    running_corrects_label += torch.sum(preds == labels.data).data.item()

  val = running_corrects_label / float(len(test.dataset))
  print(val)
  return val

### Test accuracy when using a single modality (RGB or depth)
def test_acc(net, test, DEVICE, tipo = "rgb"):
  net.train(False) 
  running_corrects_label = 0
  contatore = 0

  for rgb, depth, labels in tqdm(test):
    if tipo == "rgb":
      img = rgb.to(DEVICE)
    else:
      img = depth.to(DEVICE)
    labels = labels.to(DEVICE)

    lab = net(img)
    
    _, preds = torch.max(lab.data, 1)

    running_corrects_label += torch.sum(preds == labels.data).data.item()

  val = running_corrects_label / float(len(test.dataset))
  print(val)
  return val

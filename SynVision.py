from torchvision.datasets import VisionDataset
import re
import torchvision.transforms.functional as TF
from PIL import Image
import os
import os.path
import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt

# definition class to read images and labels
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# SYN RELATIVE ROTATION: class to load images used for the relative rotation task
class Syn_rel_rot(VisionDataset):
    #split: labels associated to the images
    #setup: problem's setup 
    def __init__(self, root, split='', setup=None, transform=None, target_transform=None):
        super(Syn_rel_rot, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.transform = transform
        self.labels = []
        self.array = []
        self.setup = setup 
        self.angles_list = [0, 90, 180, 270]

        for val in split:  
          path_rgb = val.split(" ")[0]
          
          # retrieve the correct paths for the photos
          if root == "synROD": 
            path_rgb = "/content/synROD/" + path_rgb
            path_rgb = re.sub(r'\*\*\*', 'rgb', path_rgb)
            path_depth = re.sub(r'rgb', 'depth', path_rgb)
          else: # ROD
            path_rgb = "/content/ROD/" + path_rgb
            path_rgb = re.sub(r'\?\?\?','rgb', path_rgb)
            path_rgb = re.sub(r'\*\*\*', 'crop', path_rgb)
            path_depth = re.sub(r'ROD_rgb', 'ROD_surfnorm', path_rgb)
            path_depth = re.sub(r'crop', 'depthcrop', path_depth)
        
          if os.path.isfile(path_depth) == False:
            #print(path_depth)
            continue;
            
          self.array.append((path_rgb, path_depth))
          self.labels.append( int(val.split(" ")[1]) )

        self.labels = np.array(self.labels)
        self.array = np.array(self.array)

    def __getitem__(self, index):

        image_rgb = pil_loader(self.array[index][0])
        image_depth = pil_loader(self.array[index][1])
        label = self.labels[index]
        
        #Resize the image
        image_rgb = TF.resize(image_rgb, 256)
        image_depth = TF.resize(image_depth, 256)
        
        #Horizontal flip
        if random.random() > 0.5:
            image_rgb = TF.hflip(image_rgb)
            image_depth = TF.hflip(image_depth)
            #print("Horizontal flipped") # test

        # if setup == rotation_regr -> pick two random angles between 0 and 360 
        # Then compute the difference f_angle
        if self.setup == "rotation_regr":
          angle_rgb = random.uniform(0.0, 360.0)
          angle_depth = random.uniform(0.0, 360.0)
          f_angle = (angle_depth - angle_rgb)*2*np.pi/360
        else:
          # if setup == standard -> pick two random angles from [90, 180, 270, 360]
          angle_rgb = random.choice(self.angles_list)
          angle_depth = random.choice(self.angles_list)
          delta_angle = angle_depth - angle_rgb
          if delta_angle < 0:
            delta_angle += 360
          label_angle = self.angles_list.index(delta_angle) 

        #Rotate the images given the two angles picked before
        image_rgb = TF.rotate(image_rgb, angle_rgb)
        image_depth = TF.rotate(image_depth, angle_depth)
        
        #Retrieve indices for the random crop, we apply the same random crop to both images
        crop_indices = torchvision.transforms.RandomCrop.get_params(image_rgb, output_size=(224, 224))
        i, j, h, w = crop_indices
        image_rgb = TF.crop(image_rgb, i, j, h, w)
        image_depth = TF.crop(image_depth, i, j, h, w)
        #print("Crop indices {} {} {} {}".format(i, j, h, w)) #test

        if self.transform is not None:
          image_rgb = self.transform(image_rgb)
          image_depth = self.transform(image_depth)
          
        
        # if setup == standard -> we have two labels: the relative rotation angle and the object class 
        if self.setup == "standard":
          return image_rgb, image_depth, label_angle, label
        
        # if setup == rotation_regr we have three labels the cos and sine of the relative rotation angle and the object class 
        return image_rgb, image_depth, np.cos(f_angle), np.sin(f_angle), label

    def __len__(self):
      length = len(self.array)
      return length
      
      
# SYN ABSOLUTE ROTATION, this class is very similar to Syn_rel_rot, we highlight the differences     
class Syn_abs_rot(VisionDataset):
    #split: labels associated to the images
    #setup: problem's setup 
    def __init__(self, root, split='', transform=None, target_transform=None):
        super(Syn_abs_rot, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.transform = transform
        self.labels = []
        self.array = []

        for val in split:  
          path_rgb = val.split(" ")[0]
          
          if root == "synROD": 
            path_rgb = "/content/synROD/" + path_rgb
            path_rgb = re.sub(r'\*\*\*', 'rgb', path_rgb)
            path_depth = re.sub(r'rgb', 'depth', path_rgb)
          else: # ROD
            path_rgb = "/content/ROD/" + path_rgb
            path_rgb = re.sub(r'\?\?\?','rgb', path_rgb)
            path_rgb = re.sub(r'\*\*\*', 'crop', path_rgb)
            path_depth = re.sub(r'ROD_rgb', 'ROD_surfnorm', path_rgb)
            path_depth = re.sub(r'crop', 'depthcrop', path_depth)
        
          if os.path.isfile(path_depth) == False:
            #print(path_depth)
            continue;
            
          self.array.append((path_rgb, path_depth))
          self.labels.append( int(val.split(" ")[1]) )

        self.labels = np.array(self.labels)
        self.array = np.array(self.array)

    def __getitem__(self, index):

        image_rgb = pil_loader(self.array[index][0])
        image_depth = pil_loader(self.array[index][1])
        label = self.labels[index]
        
        image_rgb = TF.resize(image_rgb, 256)
        image_depth = TF.resize(image_depth, 256)
        
        if random.random() > 0.5:
            image_rgb = TF.hflip(image_rgb)
            image_depth = TF.hflip(image_depth)
            #print("Horizontal flipped") # test

        # we pick just ONE random angle 
        angle = random.uniform(0.0, 360.0)

        # we rotate both images by the same angle
        image_rgb = TF.rotate(image_rgb, angle)
        image_depth = TF.rotate(image_depth, angle)
        
        crop_indices = torchvision.transforms.RandomCrop.get_params(image_rgb, output_size=(224, 224))
        i, j, h, w = crop_indices
        image_rgb = TF.crop(image_rgb, i, j, h, w)
        image_depth = TF.crop(image_depth, i, j, h, w)
        #print("Crop indices {} {} {} {}".format(i, j, h, w)) #test

        if self.transform is not None:
          image_rgb = self.transform(image_rgb)
          image_depth = self.transform(image_depth)
          
        # in this case we return the cosine and sine of the rotation angle applied to both images
        return image_rgb, image_depth, np.cos(angle), np.sin(angle), label

    def __len__(self):
      length = len(self.array)
      return length
    
    
    
    
    
# SYN NO ROTATION: class used to create loader with out rotation
class Syn_no_rotation(VisionDataset):
    #split: labels associated to the images
   
    def __init__(self, root, split='', setup=None, transform=None, target_transform=None, tipo = None):
        super(Syn_no_rotation, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.transform = transform
        self.labels = []
        self.array = []
        self.tipo = tipo

        for val in split:  
          path_rgb = val.split(" ")[0]
          if root == "synROD": 
            path_rgb = "/content/synROD/" + path_rgb
            path_rgb = re.sub(r'\*\*\*', 'rgb', path_rgb)
            path_depth = re.sub(r'rgb', 'depth', path_rgb)
          else: # ROD
            path_rgb = "/content/ROD/" + path_rgb
            path_rgb = re.sub(r'\?\?\?','rgb', path_rgb)
            path_rgb = re.sub(r'\*\*\*', 'crop', path_rgb)
            path_depth = re.sub(r'ROD_rgb', 'ROD_surfnorm', path_rgb)
            path_depth = re.sub(r'crop', 'depthcrop', path_depth)
          
          if os.path.isfile(path_depth) == False:
            #print(path_depth)
            continue;

          self.array.append((path_rgb, path_depth ))
          self.labels.append( int(val.split(" ")[1]) )

        self.labels = np.array(self.labels)
        self.array = np.array(self.array)

    def __getitem__(self, index):

        image_rgb = pil_loader(self.array[index][0])
        image_depth = pil_loader(self.array[index][1])
        label = self.labels[index]

        image_rgb = TF.resize(image_rgb, 256)
        image_depth = TF.resize(image_depth, 256)
        
        # if self == train -> apply RandomCrop and horizontal flip
        if self.tipo != "test":
            crop_indices = torchvision.transforms.RandomCrop.get_params(image_rgb, output_size=(224, 224))
            i, j, h, w = crop_indices
            image_rgb = TF.crop(image_rgb, i, j, h, w)
            image_depth = TF.crop(image_depth, i, j, h, w)
            if (random.random() > 0.5):
                image_rgb = TF.hflip(image_rgb)
                image_depth = TF.hflip(image_depth)
        else: # self == test -> apply center crop
            image_rgb = TF.center_crop(image_rgb, 224)
            image_depth = TF.center_crop(image_depth, 224)
            

        if self.transform is not None:
          image_rgb = self.transform(image_rgb)
          image_depth = self.transform(image_depth)
        
        return image_rgb, image_depth, label
        

    def __len__(self):
      length = len(self.array)
      return length
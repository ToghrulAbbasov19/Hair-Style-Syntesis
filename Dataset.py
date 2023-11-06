import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision import datasets

class MyRefDataset(Dataset):
  def __init__(self, reference_data, reference_label, transform):
    super().__init__()
    self.ref_data=reference_data
    self.transform=transform
    self.ref_label=reference_label
    self.ref_data2=random.sample(reference_data, len(reference_data))


  def __len__(self):
    return len(self.ref_data)

  def __getitem__(self, index):
    image1=self.ref_data[index]
    image2=self.ref_data2[index]
    label=self.ref_label[index]

    image1=Image.open(image1).convert("RGB")
    image2=Image.open(image2).convert("RGB")
    if self.transform is not None:
      # augmented_images=self.transform(image=image1, image1=image2)
      image1=self.transform(image1)
      image2=self.transform(image2)
    image1=image1.float()
    image2=image2.float()
    return image1, image2, label


transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5]), # mean and std can be changed
])
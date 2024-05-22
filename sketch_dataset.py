from torch.utils.data.dataset import Dataset
import os
from PIL import Image

class SketchDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root
        #print(self.root)
        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        # root / <label>  /  <item> / <view>.png
        for label in os.listdir(root): # Label
            for sketch in os.listdir(root + '/' + label):

                self.x.append(root + '/' + label + '/' + sketch)
                self.y.append(self.class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_sketch = self.x[index]
        path = orginal_sketch.split('/')[-2]+'/'+orginal_sketch.split('/')[-1]
        im = Image.open(orginal_sketch)
        im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)

        return im, self.y[index],path

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
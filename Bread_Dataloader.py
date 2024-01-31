import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pycocotools.coco import COCO
import yaml
import cv2
from torchvision import transforms

class BreadDataset(Dataset):
    def __init__(self, cfg_path, transform=None, train = True):
          
        with open(cfg_path, 'r', encoding='utf-8') as file:
            self.cfg = yaml.safe_load(file)
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.data_path = self.cfg['data']['data_path']
        
        if train == True:
            self.img_dir = self.data_path + "train/"
        else:
            self.img_dir = self.data_path + "test/"

        annotations_path =self.img_dir + "_annotations.coco.json"
        self.coco = COCO(annotations_path)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.image_ids = self.coco.getImgIds()
        self.images = self.coco.loadImgs(self.image_ids)


    def __len__(self):
        return len(self.image_ids )
    

    def __getitem__(self, idx):     
            img_id = self.image_ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.coco.getCatIds(), iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            image_info = self.coco.loadImgs(img_id)[0]
            image_path = os.path.join(self.img_dir, image_info['file_name'])
            image = Image.open(image_path)
            image = self.transform(image)
            boxes = []
            classes = []
            for ann in anns:
                x_min, y_min, width, height = ann['bbox']
                boxes.append([x_min, y_min, x_min + width, y_min + height])  
                classes.append(ann['category_id'])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            return img_id, image, boxes, classes 
        



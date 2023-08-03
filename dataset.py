import os
import numpy as np
import torch
from PIL import Image


class DetectDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv, transform=None):
        super(DetectDataset, self).__init__()
        self.transform = transform
        self.img_dir = os.path.join(img_dir, 'training_images')
        self.label_file = csv

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.label_file.loc[idx, 'image'])
        image = Image.open(img_path)
        image = np.array(image)
        image_id = torch.tensor([idx])
        lbl = self.label_file.loc[self.label_file['image'] == self.label_file.loc[idx, 'image']]
        boxes = lbl[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        target = dict()
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, category_ids=labels)
            image, boxes, _ = transformed["image"], transformed["bboxes"], transformed["category_ids"]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target["image_id"] = image_id
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["labels"] = labels
        return image, target

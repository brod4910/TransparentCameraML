import torch.utils.data as data
from PIL import Image
import os
import os.path
from pycocotools.coco import COCO
import torch

CLASSES = {'person':0,'vehicle':1,'outdoor':2,'animal':3,
'accessory':4,'sports':5,'kitchen':6,'food':7,
'furniture':8,'electronic':9,'appliance':10,'indoor':11}

class CocoDataset(data.Dataset):
	def __init__(self, root, annotation_file, transform= None):
		self.root = root
		self.coco = COCO(annotation_file)
		self.ids = list(self.coco.imgs.keys())
		self.transform = transform

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, idx):
		coco = self.coco
		img_id = self.ids[idx]
		ann_ids = coco.getAnnIds(imgIds= img_id)
		target = coco.loadAnns(ann_ids)

		cat_ids = [targ['category_id'] for targ in target]
		cats = coco.loadCats(ids=cat_ids)

		sup_cats = [cat['supercategory'] for cat in cats]

		labels = []

		for label in sup_cats:
			l = CLASSES[label]
			if l not in labels:
				labels.append(l)

		path = coco.loadImgs(img_id)[0]['file_name']

		img = Image.open(os.path.join(self.root, path)).convert('RGB')

		if self.transform is not None:
			img = self.transform(img)

		return img, labels
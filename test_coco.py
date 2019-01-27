from cocoapi.PythonAPI.pycocotools.coco import COCO
import torch
import torchvision.datasets as datasets
from Models.normalize import CastTensor
from torchvision import transforms
from gluoncv import data, utils
from Models.CocoDataset import CocoDataset

def main():
    dataDir ='/Volumes/MyPassport/coco'
    dataType ='val2017'
    annFile ='{}/annotations/instances_{}.json'.format(dataDir, dataType)

    # coco = COCO(annFile)

    # cats = coco.loadCats(coco.getCatIds())

    # nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    val_dataset = CocoDataset('/Volumes/MyPassport/coco/val2017', annFile, transform=transforms.ToTensor())
    
    val_dataloader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size= 1, 
    shuffle= True, 
    num_workers= 2,
    pin_memory= True
    )

    for i, (input, raw_labels) in enumerate(val_dataloader):
        if i == 3:
            break
        # print(raw_labels)
        labels = [raw.item() for raw in raw_labels]
        labels = torch.as_tensor(labels)

        # print("Batch index: {}".format(i))
        # labels = [label for label in raw_labels]
        # print(labels, len(labels))
        # cat_ids = [label['category_id'] for label in labels if 'category_id' in label]
        # print(cat_ids)
        # cats = val_dataset_torch.coco.loadCats(ids=cat_ids)
        # print(cats)
        # sup_cats = cats[:]['supercategory']
        # print(sup_cats)

if __name__ == '__main__':
    main()

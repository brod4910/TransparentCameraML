from cocoapi.PythonAPI.pycocotools.coco import COCO
import torch
import torchvision.datasets as datasets
from Models.normalize import CastTensor
from torchvision import transforms
from gluoncv import data, utils

def main():
    dataDir ='/Volumes/MyPassport/coco/'
    dataType ='val2017'
    annFile ='{}/annotations/instances_{}.json'.format(dataDir, dataType)

    # coco = COCO(annFile)

    # cats = coco.loadCats(coco.getCatIds())

    # nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))


    d = transforms.Compose([transforms.Resize((50,50)),
    CastTensor()])

    # train_dataset = data.COCODetection(root='/Volumes/MyPassport/coco', splits=['instances_train2017'])
    val_dataset = data.COCODetection(root='/Volumes/MyPassport/coco', splits=['instances_val2017'])
    
    # print('Num of training images:', len(train_dataset))
    print('Num of validation images:', len(val_dataset))

    val_image, val_label = val_dataset[0]

    print(val_label)


    # coco = datasets.CocoDetection(root= '/Volumes/MyPassport/coco/val2017/val2017', annFile= annFile, transform= d)

    # train_loader = torch.utils.data.DataLoader(
    # coco,
    # batch_size= 1, 
    # shuffle= True, 
    # num_workers= 2,
    # pin_memory= True
    # )

    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    #     # for pixel in iter(inputs.getdata()):
    #     #     print(pixel)
    #     # print(targets)
    #     if not targets:
    #         print('None hit')
    #         print(inputs)
    #         print(batch_idx)
    #         continue
    #     cat_ids = targets[0]['category_id']
    #     targs = coco.coco.loadCats(ids= [cat.item() for cat in cat_ids])
    #     # print([targ['supercategory'] for targ in targs])
    #     # break

if __name__ == '__main__':
    main()





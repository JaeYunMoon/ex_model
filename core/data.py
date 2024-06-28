import os 
import glob
import itertools
from pathlib import Path
import numpy as np 
from PIL import Image
from munch import Munch

import torch 
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder


"""
segmentation 이미지를 어떻게 Noramlize 할 것인가? 
pix2pix 는 그냥 rgb로 받아 옴 

segmentation image를 어떻게 const vector 로 변경 할 것인가 
    1. conv2d 를 사용해서
    2. 그냥 mlp를 사용해서 

    - 이미지를 분석하는 것이 아니기 때문에 conv2d로 할 필요가 없지 않나 06/26

style GAN은 라벨 자체가 필요없어서, image만 불러온다. 
star GAN v2은 라벨(도메인)이 필요하여, torchvision.datasets.ImageFolder로 불러온다.
    
*실제와 가상의 라벨이 필요하고, segmentation은 필요할 듯 함 (확정 아님)
    - 실제와 가상의 라벨이 필요한가? , segmentation을 꼭 써야하나? 
    - 큰 이미지에서 정확한 이미지가 나올려나? -> size를 줄이자. 


"""

IMG_SUFFIX = ["png",'jpeg','jpg','JPG']
# def listdir(dname):
#     fnames,segnames = [],[] 

#     suf_f = list(Path(dname).rglob('*.'+suf) for suf in IMG_SUFFIX)
#     if len(suf_f) != 0:
#         for i in suf_f:
#             seg_im = i.replace("img","seg")
#             fnames.append(i)
#             segnames.append(i)
#     return fnames,segnames

def listdir(dname):
    fnames = list(itertools.chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def _transforms(imgsz=None,c_mean=(0.5,0.5,0.5),c_std=(0.5,0.5,0.5)):
    """
    imgsz = args 인자로 받아야함 
    """

    if imgsz is not None:
        transform = transforms.Compose([
            transforms.Resize([imgsz,imgsz]),
            transforms.ToTensor(),
            transforms.Normalize(mean=c_mean,std=c_std)
        ])
    
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=c_mean,std=c_std)
        ])

    return transform

def get_seg_dataloader(fdir,imgsz):
    """
    가상 데이터 즉, 변환하고자 하는 데이터의 segmentation의 데이터를 가져오는 함수 
    - content의 위치를 그대로 사용하기 위해, 
    - segmentation 라벨을 그대로 사용하기 위해 
    """
    pass 

def get_dir_dataloader(fdir,batch_size,shuffle=True,
                       num_workers=4,
                       imgsz=None,custom_mean=(.5,.5,.5),custom_std=(.5,.5,.5)):
    """
    이미지 폴더에 있는 데이터를 dataLoader로 만들어 줌 
    input = dir, imgsz,custom_mean,custom_std -> custom 붙은 것은 transforms에 들어갈 파라미터 
    output = DataLoader -> img,domain_label 
    """
    transform = _transforms(imgsz,custom_mean,custom_std)
    dataset = ImageFolder(fdir,transform)
    sampler = _make_balanced_sampler(dataset.targets) # 편향된 데이터를 batch 작성 할 때 고려해서 작성해주는 파라미터 
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      sampler=sampler, 
                      drop_last=True,
                      pin_memory=True)

def get_seg_dataloader(fdir,batch_size,
                       shuffle = True,num_workers =4,
                       imgsz=None,c_mean=(0.5,0.5,0.5),c_std=(0.5,0.5,0.5)):
    """
    sythetic image의 segmentation이 있는 directory 주소 
    """
    dataset = CustomSegDataset(fdir,imgsz=imgsz,
                               transeform_bool=True,
                               custom_mean=c_mean,custom_std=c_std)
    
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=True,
                      pin_memory=True
                      )




class Custom_sythetic_add_segDataset(Dataset):
    """
    Segmentation image와 원본 이미지들의 라벨과 이미지,
      동시에 가상 데이터의 segmentation과 image는 동일하게 가져오는 클래스 -> 사용 안할 듯 
    """
    def __init__(self,imgsz:int,dataset_dir:str,
                 transeform_bool = True,
                 custom_mean=(0.5,0.5,0.5),custom_std=(0.5,0.5,0.5),
                 custom_seg_mean=(0.5,0.5,0.5),custom_seg_std=(0.5,0.5,0.5),) -> None:
        self.transeform_bool = transeform_bool
        self.transeform = _transforms(imgsz,custom_mean,custom_std)
        self.seg_transeform = _transforms(imgsz,custom_seg_mean,custom_seg_std)
        self.img_match,self.target = self._make_dataset(dataset_dir)
    
    def _make_dataset(self,root):
        domain = os.listdir(root)
        flist,flist2,labels = [],[],[]  
        for idx,dm in enumerate(sorted(domain)):
            class_dir = os.path.join(root,dm)
            class_fname,seg_class_fname = listdir(class_dir)
            flist += class_fname
            flist2 +=seg_class_fname
            labels += [idx]*len(class_fname)
        assert len(flist) == len(flist2) == len(labels),"Segmentation image and original image and label do not match"
        return list(zip(flist,flist2)),labels 
    
    def __getitem__(self, index):
        fname,fname2 = self.img_match[index]
        label = self.target[index]
        im = Image.open(fname)
        sim = Image.open(fname)
        if self.transeform_bool:
            sim = self.seg_transeform(sim)
            im = self.transeform(im)
        return im,sim,label 
    
    def __len__(self):
        return len(self.target)

class CustomSegDataset(Dataset):
    def __init__(self,dataset_dir:str,imgsz:int,
                 transeform_bool=True,
                 custom_mean=(0.5,0.5,0.5),custom_std=(0.5,0.5,0.5)) -> None:
        self.transform_bool = transeform_bool
        self.transform= _transforms(imgsz,custom_mean,custom_std)
        self.img = self._make_dataset(dataset_dir)

    def _make_dataset(self,root):
        seg_ims = listdir(root)
        
        return seg_ims
    def __getitem__(self, index):
        im_f = self.img(index)
        img  = Image.open(im_f)
        if self.transform_bool:
            img = self.transform(img)
        return img
            

class InputFetcher:
    def __init__(self,loader,segloader=None,lataten_dim=16,mode='') -> None:
        self.loader = loader
        self.seg_loader = segloader
        self.latent_dim = lataten_dim
        self.mode = mode 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_inputs(self):
        try:
            x,y = next(self.iter)
        except (AttributeError,StopIteration):
            self.iter = iter(self.loader)
            x,y = next(self.iter)
    
        return x,y
    
    def _fetch_inputs_seg(self):
        try:
            s_x = next(self.seg_loader)
        except (AttributeError,StopIteration):
            self.iter = iter(self.seg_loader)
            s_x = next(self.seg_loader)
        return s_x 
    
    def __next__(self):
        x,y = self._fetch_inputs()
        if self.mode == 'train':
            x_seg = self._fetch_inputs_seg()
            z_trg = torch.randn(x.size(0),self.latent_dim)
            z_trg2 = torch.randn(x.size(0),self.latent_dim)
            inputs = Munch(x=x,y=y,x_seg=x_seg,z_trg=z_trg,z_trg2 = z_trg2)

        elif self.mode =='val':
            x_seg = self._fetch_inputs_seg()
            inputs = Munch(x=x,y=y,x_seg =x_seg)

        elif self.mode =="test":
            inputs = Munch(x=x,y=y)

        else:
            raise NotImplementedError
        
        return Munch({k:v.to(self.device) for k,v in inputs.items()})



    
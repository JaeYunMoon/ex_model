import os 
import argparse 
import torch 
from munch import Munch
from core.data import get_dir_dataloader,get_seg_dataloader

def main(args):
    print(args)
    torch.manual_seed(args.seed)
    
    #slover=Solver(args)

    if args.mode=='train':
        loaders = Munch(src = get_dir_dataloader(fdir = args.train_dir,
                                                batch_size=args.batch_size,
                                                imgsz=args.imgsz,
                                                custom_mean=args.image_transefomer_mena,
                                                custom_std=args.image_transefomer_std,
                                                num_workers=args.num_workers),
                        seg = get_seg_dataloader(fdir=args.seg_dir,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 imgsz=args.imgsz,
                                                 custom_mean=args.seg_image_transefomer_mean,
                                                 custom_std=args.seg_image_transerfomer_std
                                                 ),
                        val = get_dir_dataloader(fdir=args.val_dir,
                                                 batch_size=args.batch_size,
                                                 imgsz=args.imgsz,
                                                 num_workers=args.num_workers,
                                                 custom_mean=args.image_transefomer_mean,
                                                 custom_std=args.image_transefomer_std)
                        )
        #

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='sim2real')
    
    # directory 
    parser.add_argument('--train_dir',type=str,default="datasets/*",
                        help='Directory containing training images')
    parser.add_argument('--seg_dir',type=str,default="datasets/*")
    
    parser.add_argument('--val_dir',type=str,default="datasets/*",
                        help='Directory containing Validation images')
    parser.add_argument('--seg_image_transefomer_mean',type=tuple,default=(.5,.5,.5))
    parser.add_argument('--seg_image_transefomer_std',type=tuple,default=(.5,.5,.5)) 
    parser.add_argument('--image_tarnsefomer_mean',type=tuple,default=(.5,.5,.5))
    parser.add_argument('--image_transefomer_std',type=tuple,default=(.5,.5,.5))
    parser.add_argument('--batch_size',type=int,default=8)

    parser.add_argument('--seed',type=int,default=1993,help='seed for random number generator')


    parser.add_argument('--latent_dim',type=int,default=16)
    parser.add_argument('--imgsz',type=int,default=256)

    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints/',
                        help='Directory for saving network checkpoints')
    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    
    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval/',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

   # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results/',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src/',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref/',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female/',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female/',
                        help='output directory when aligning faces')


    args = parser.parse_args()
    main(args)

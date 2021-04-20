import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
from tqdm import tqdm, trange
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--dataset_dir', metavar='DIR', required=True, help='dataset directory for evaluation')
parser.add_argument('--model_path', metavar='DIR', required=True, help='pre-trained model (.pth) path')
# parser.add_argument('--depth_dir', metavar='DIR', required=True, help='depth data directory')
opt = parser.parse_args()

dataset_path = opt.dataset_dir
# depth_path = opt.depth_dir

generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load(opt.model_path))

generator.cuda()
generator.eval()

test_datasets = ['DES', 'LFSD','NJU2K','NLPR','SIP','STERE']
#test_datasets = ['STERE']

for _, dataset in enumerate(tqdm(test_datasets, desc='1st loop')):
    save_path = os.path.join(dataset_path, 'results', dataset) # './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = os.path.join(dataset_path, dataset, 'RGB') # dataset_path + dataset + '/RGB/'
    depth_root = os.path.join(dataset_path, dataset, 'depth') # dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, depth_root, opt.testsize)
    for i in trange(test_loader.size, desc='2nd loop'):
        # print i
        image, depth, HH, WW, imgname = test_loader.load_data()
        image = image.cuda()
        depth = depth.cuda()
        generator_pred = generator.forward(image, depth, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze() * 255.0
        # print(np.min(res))
        # print(np.max(res))
        # misc.imsave(os.path.join(save_path, imgname), res)
        cv2.imwrite(os.path.join(save_path, imgname), res)

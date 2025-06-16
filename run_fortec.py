import argparse
import os
from model import RFRNetModel
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',default='./test/', type=str)
    # parser.add_argument('--data_root',default='./modeltest/', type=str)
    parser.add_argument('--mask_root', default='./mask/',type=str)
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=80)
    parser.add_argument('--mask_mode', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=500000)
    parser.add_argument('--model_path', type=str, default="./checkpoint/g_200000.pth")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--finetune',action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()



    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()
    if args.test:
        model.initialize_model(args.model_path, False)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, training=False))
        model.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path, True)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     model = nn.DataParallel(model)
        # model.to(device)
        model.cuda()

        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        model.train(dataloader, args.model_save_path, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()
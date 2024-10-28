import argparse
import os
import rootutils
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torcheval.metrics import PeakSignalNoiseRatio


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logger import Logger
from src.model.sr_net import SRNet
from src.dataloader.div2k import DIV2K



def test(args):
    device = args.device
    # Definim el model
    model = SRNet(sampling=args.sampling,kernel_size=args.kernel_size, features=args.features, blocks=args.blocks).to(device)
    # Carragam els pesos en el model
    weights = torch.load(args.ckpt_path,map_location=args.device)
    model.load_state_dict(weights['model_state_dict'])
    # Carragam els dataloader
    test_dataset = DIV2K(sampling=args.sampling,subset="test",path=args.dataset_path, patch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False, num_workers=8)
    # Instancia el monitoritzador de la mètrica
    PSNR = PeakSignalNoiseRatio(data_range=1.)
    # Crea la carpeta on guardar les imatges
    os.makedirs(args.output_path,exist_ok=True)
    for i, batch in enumerate(tqdm(test_loader)):
        gt, low = batch
        gt = gt.to(device)
        low = low.to(device)
        # Utilitzam torch.no_grad() per no utilitzar tanta memoria i poder fer els calculs més ràpid.
        with torch.no_grad():
            high = model(low)
        # Guarda la imatge i actualitza el monitoritzador de la mètrica
        save_image(high, f'{args.output_path}/{i}_out.png')
        save_image(low, f'{args.output_path}/{i}_low.png')
        bic = nn.functional.interpolate(low, scale_factor=args.sampling, mode='bicubic')
        save_image(bic, f'{args.output_path}/{i}_bic.png')
        PSNR.update(high, gt)
    # Calcla la mitjana de la mètrica i l'escriu per pantalla
    test_psn_mean = PSNR.compute()
    print(f"The mean PSNR over the test set is {test_psn_mean}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument('--sampling', type=int, required=True, help='Sampling factor')
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path of the weights checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path where save the output images")

    parser.add_argument("--kernel_size", type=int,default=3,help="Kernel size for 2d convoltion")
    parser.add_argument("--features", type=int, default=64, help="Number of features for residual blocks")
    parser.add_argument("--blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train the model")


    args = parser.parse_args()

    test(args)




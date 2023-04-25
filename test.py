import torchvision
import torch
import kornia
from UNet import UNet32, UNet
import argparse
from kornia.metrics import psnr
from torch.utils.tensorboard import SummaryWriter
from ptcolor import rgb2lab, deltaE94
from train import fivek


def parse_args():
    parser = argparse.ArgumentParser("i2i Parser")
    a = parser.add_argument
    a("exp_name", help="name of the experiment")
    a("-s", "--size", type=int, default=32, help="image size")
    return parser.parse_args()


def test():
    args = parse_args()
    basedir = args.basedir
    raw_dir = os.path.join(basedir, 'raw')
    exp_dir = os.path.join(basedir, 'expC/')
    file_test = os.path.join(basedir, 'test-list.txt')
    net = UNet32(0.0).to(DEVICE)
    weights = torch.load(os.path.join(basedir, './ckt/best_' + args.exp_name + '.pt'))
    net.load_state_dict(weights)
    val_set = fivek(FILE_TEST, RAW_DIR, EXP_DIR, args.size, False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)
    psnrs = []
    val_losses = []
    net.eval()
    for img, gt, idx in val_loader:
        img, gt = img.to(DEVICE), gt.to(DEVICE)
        pred = net(img)
        loss_psnr = psnr(pred, gt, 1.0).mean()
        psnrs.append(loss_psnr.item())
        val_losses.append(loss.mean().item())
        to_save = ((pred.squeeze(0)) * 255).to(torch.uint8)
        torchvision.io.write_png(to_save.to('cpu'), BASE_DIR + 'experiments/' + val_set.files[idx])
    print('LPIPS', sum(val_losses) / len(val_losses))
    print('PSNR', sum(psnrs) / len(psnrs))


if __name__ == '__main__':
    test()

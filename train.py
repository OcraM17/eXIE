import torchvision
import torch
from UNet import UNet32, UNet
import argparse
import os
from kornia.metrics import psnr
from torch.utils.tensorboard import SummaryWriter
from ptcolor import rgb2lab, deltaE94
import torchvision.transforms.functional as TF
import random


def parse_args():
    parser = argparse.ArgumentParser("i2i Parser")
    a = parser.add_argument
    a("--exp_name", help="name of the experiment")
    a("--basedir", help="basedir")
    a("-s", "--size", type=int, default=32, help="image size")
    a("-e", "--epochs", type=int, default=600, help="Number of Epochs")
    a("-l", "--lr", type=float, default=1e-3, help="Learning Rate")
    a("-b", "--batch-size", type=int, default=30, help="Size of the mini batches")
    a("-d", "--dropout", type=float, default=0.0, help="Dropout value")
    return parser.parse_args()


class MyRotationTransform:

    def __init__(self):
        self.angles = [-90, 0, 90]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class fivek(torch.utils.data.Dataset):
    def __init__(self, file, raw_dir, exp_dir, size, train=True):
        super(fivek, self).__init__()
        self.file = file
        self.raw_dir = raw_dir
        self.exp_dir = exp_dir
        self.files = []
        self.train = train
        self.size = size
        self.process()

    def process(self):
        with open(self.file) as f:
            for line in f:
                l = line.split()[0]
                self.files.append(l)
        print(self.__len__())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        raw = torchvision.io.read_image(self.raw_dir + self.files[idx]).float() / 255.0
        exp = torchvision.io.read_image(self.exp_dir + self.files[idx]).float() / 255.0
        r_e = torch.stack([raw, exp], 0)
        if self.train:
            trasf = torchvision.transforms.Compose(
                [torchvision.transforms.RandomHorizontalFlip(0.5), torchvision.transforms.RandomVerticalFlip(0.5),
                 MyRotationTransform(), torchvision.transforms.RandomResizedCrop((self.size, self.size))])
            r_e = trasf(r_e)
            return r_e[0], r_e[1]

        if not self.train:
            r_e = torchvision.transforms.Resize((self.size, self.size))(r_e)
            return r_e[0], r_e[1], idx


def train(net, train_loader, writer, optimizer, losses, epoch, bce):
    for img, gt in train_loader:
        img, gt = img.to(DEVICE), gt.to(DEVICE)
        optimizer.zero_grad()
        pred = net(img)
        loss_bce = bce(pred, gt).mean()
        loss_psnr = -psnr(pred, gt, 1.0).mean()
        loss_delta = deltaE94(rgb2lab(pred), rgb2lab(gt)).mean()
        losses[0].append(loss_delta.item())
        losses[1].append(-loss_psnr.item())
        loss_bce.backward()
        optimizer.step()
        if len(losses[0]) % 10 == 0:
            print('TRAIN PSNR', sum(losses[1]) / len(losses[1]), len(losses[1]))
            writer.add_scalar('TRAIN PSNR', sum(losses[1]) / len(losses[1]), len(losses[1]))
            writer.add_scalar('TRAIN DeltaE', sum(losses[0]) / len(losses[0]), len(losses[1]))


def valid(net, val_loader, writer, epoch):
    psnrs, deltaEs = [], []
    for img, gt, _ in val_loader:
        img, gt = img.to(DEVICE), gt.to(DEVICE)
        pred = net(img)
        loss_psnr = psnr(pred, gt, 1.0).mean()
        loss_delta = deltaE94(rgb2lab(pred), rgb2lab(gt)).mean()
        deltaEs.append(loss_delta.item())
        psnrs.append(loss_psnr.item())
    print('VAL PSNR', sum(psnrs) / len(psnrs), epoch)
    writer.add_scalar('VAL PSNR', sum(psnrs) / len(psnrs), epoch)
    writer.add_scalar('Val DeltaE', sum(deltaEs) / len(deltaEs), epoch)
    writer.add_images('VAL images', pred[0:16], epoch)
    return sum(psnrs) / len(psnrs)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir
    raw_dir = os.path.join(basedir, 'raw')
    exp_dir = os.path.join(basedir, 'expC/')
    file = os.path.join(basedir, 'train12-list.txt')
    file_test = os.path.join(basedir, 'test-list.txt')

    writer = SummaryWriter(os.path.join(basedir, 'tensor', args.exp_name))
    net = UNet32(args.dropout).to(device)
    bce = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.AdamW(net.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600], gamma=0.1)
    train_set = fivek(file, raw_dir, exp_dir, args.size, True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_set = fivek(file_test, raw_dir, exp_dir, args.size, False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)
    losses = [[], []]
    max_psnr = 0.0
    for epoch in range(args.epochs):
        print('TRAINING EPOCH', epoch)
        net.train()
        train(net, train_loader, writer, optimizer, losses, epoch, bce)
        scheduler.step()
        net.eval()
        print('VALIDATION EPOCH', epoch)
        actual = valid(net, val_loader, writer, epoch)
        if actual > max_psnr:
            max_psnr = actual
            best_model = net.state_dict()
            torch.save(best_model, './ckt/best_' + args.exp_name + '.pt')


if __name__ == '__main__':
    main()

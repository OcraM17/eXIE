import torch


class Encoder_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_Block, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, (4, 4), stride=2, padding=(1, 1))
        self.batch = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act(self.batch(self.conv(x)))
        return x


class Decoder_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dv):
        super(Decoder_Block, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, (4, 4), stride=2, padding=(1, 1))
        self.batch = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dv)

    def forward(self, x, c):
        x = self.drop(self.batch(self.conv(x)))
        x = torch.cat([x, c], dim=1)
        x = self.act(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = Encoder_Block(3, 64)
        self.encoder2 = Encoder_Block(64, 128)
        self.encoder3 = Encoder_Block(128, 256)
        self.encoder4 = Encoder_Block(256, 512)
        self.encoder5 = Encoder_Block(512, 512)
        self.encoder6 = Encoder_Block(512, 512)
        self.encoder7 = Encoder_Block(512, 512)

        self.bottle = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, (4, 4), 2, padding=(1, 1)),
            torch.nn.ReLU()
        )

        self.decoder1 = Decoder_Block(512, 512)
        self.decoder2 = Decoder_Block(1024, 512)
        self.decoder3 = Decoder_Block(1024, 512)
        self.decoder4 = Decoder_Block(1024, 512)
        self.decoder5 = Decoder_Block(1024, 256)
        self.decoder6 = Decoder_Block(512, 128)
        self.decoder7 = Decoder_Block(256, 64)

        self.output = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 3, (4, 4), 2, padding=(1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)

        bottle = self.bottle(e7)

        d1 = self.decoder1(bottle, e7)
        d2 = self.decoder2(d1, e6)
        d3 = self.decoder3(d2, e5)
        d4 = self.decoder4(d3, e4)
        d5 = self.decoder5(d4, e3)
        d6 = self.decoder6(d5, e2)
        d7 = self.decoder7(d6, e1)

        out = self.output(d7)
        return out


class UNet32(torch.nn.Module):
    def __init__(self, dv):
        super(UNet32, self).__init__()
        self.encoder1 = Encoder_Block(3, 32)
        self.encoder2 = Encoder_Block(32, 64)
        self.encoder3 = Encoder_Block(64, 128)
        self.encoder4 = Encoder_Block(128, 128)

        self.bottle = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, (4, 4), 2, padding=(1, 1)),
            torch.nn.ReLU()
        )

        self.decoder1 = Decoder_Block(128, 128, dv)
        self.decoder2 = Decoder_Block(256, 128, dv)
        self.decoder3 = Decoder_Block(256, 64, dv)
        self.decoder4 = Decoder_Block(128, 32, dv)

        self.output = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 3, (4, 4), 2, padding=(1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        bottle = self.bottle(e4)

        d1 = self.decoder1(bottle, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)

        out = self.output(d4)
        return out


if __name__ == '__main__':
    net = UNet32(0.05)
    par = 0
    net(torch.rand(1, 3, 32, 32))
    for param in net.parameters():
        par += param.numel()
    print(par)

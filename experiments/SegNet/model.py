import torch
import torch.nn as nn
import torchvision.models as models

class SegNet(nn.Module):
    def __init__(self, num_classes=32):
        super(SegNet, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.enc1 = nn.Sequential(*vgg16.features[:6])
        self.enc2 = nn.Sequential(*vgg16.features[7:13])
        self.enc3 = nn.Sequential(*vgg16.features[14:23])
        self.enc4 = nn.Sequential(*vgg16.features[24:33])
        self.dec4 = self.decoder_block(512, 256)
        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(128, 64)
        self.dec1 = self.decoder_block(64, 64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x1p, ind1 = self.pool(x1)
        x2 = self.enc2(x1p)
        x2p, ind2 = self.pool(x2)
        x3 = self.enc3(x2p)
        x3p, ind3 = self.pool(x3)
        x4 = self.enc4(x3p)
        x4p, ind4 = self.pool(x4)
        d4 = self.unpool(x4p, ind4, output_size=x4.size())
        d4 = self.dec4(d4)
        d3 = self.unpool(d4, ind3, output_size=x3.size())
        d3 = self.dec3(d3)
        d2 = self.unpool(d3, ind2, output_size=x2.size())
        d2 = self.dec2(d2)
        d1 = self.unpool(d2, ind1, output_size=x1.size())
        d1 = self.dec1(d1)
        return self.classifier(d1)

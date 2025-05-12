# full_lbunet_with_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSkipFusion(nn.Module):
    def __init__(self, in_channels_decoder, in_channels_encoder, out_channels):
        super().__init__()
        self.align_decoder = nn.Conv2d(in_channels_decoder, out_channels, kernel_size=1)
        self.align_encoder = nn.Conv2d(in_channels_encoder, out_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, decoder_feat, encoder_feat):
        decoder_feat = self.align_decoder(decoder_feat)
        encoder_feat = self.align_encoder(encoder_feat)
        fused = torch.cat([decoder_feat, encoder_feat], dim=1)
        attention_map = self.attention(fused)
        out = decoder_feat + attention_map * encoder_feat
        return self.output_conv(out)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LBUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64]):
        super(LBUNet, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(input_channels, c_list[0])
        self.enc2 = ConvBlock(c_list[0], c_list[1])
        self.enc3 = ConvBlock(c_list[1], c_list[2])
        self.enc4 = ConvBlock(c_list[2], c_list[3])
        self.enc5 = ConvBlock(c_list[3], c_list[4])
        self.enc6 = ConvBlock(c_list[4], c_list[5])

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up5 = nn.ConvTranspose2d(c_list[5], c_list[4], kernel_size=2, stride=2)
        self.fuse5 = AttentionSkipFusion(c_list[4], c_list[4], c_list[4])
        self.dec5 = ConvBlock(c_list[4], c_list[4])

        self.up4 = nn.ConvTranspose2d(c_list[4], c_list[3], kernel_size=2, stride=2)
        self.fuse4 = AttentionSkipFusion(c_list[3], c_list[3], c_list[3])
        self.dec4 = ConvBlock(c_list[3], c_list[3])

        self.up3 = nn.ConvTranspose2d(c_list[3], c_list[2], kernel_size=2, stride=2)
        self.fuse3 = AttentionSkipFusion(c_list[2], c_list[2], c_list[2])
        self.dec3 = ConvBlock(c_list[2], c_list[2])

        self.up2 = nn.ConvTranspose2d(c_list[2], c_list[1], kernel_size=2, stride=2)
        self.fuse2 = AttentionSkipFusion(c_list[1], c_list[1], c_list[1])
        self.dec2 = ConvBlock(c_list[1], c_list[1])

        self.up1 = nn.ConvTranspose2d(c_list[1], c_list[0], kernel_size=2, stride=2)
        self.fuse1 = AttentionSkipFusion(c_list[0], c_list[0], c_list[0])
        self.dec1 = ConvBlock(c_list[0], c_list[0])

        # Output
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))

        # Decoder with AttentionSkipFusion
        d5 = self.up5(e6)
        d5 = self.fuse5(d5, e5)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = self.fuse4(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.fuse3(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.fuse2(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.fuse1(d1, e1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return torch.sigmoid(out)  # Make sure sigmoid is applied for binary output


# Updated loss function
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.wb = wb
        self.wd = wd

    def forward(self, out, target):
        out = torch.clamp(out, min=1e-7, max=1 - 1e-7)
        target = target.float()
        bce = F.binary_cross_entropy(out, target)
        smooth = 1e-5
        intersection = (out * target).sum()
        dice = (2. * intersection + smooth) / (out.sum() + target.sum() + smooth)
        return self.wb * bce + self.wd * (1 - dice)

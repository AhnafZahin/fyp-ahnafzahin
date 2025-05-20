# low_light_enhancement_model.py
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class LowLightEnhancementNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(LowLightEnhancementNet, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.dec3 = DecoderBlock(512 + 256, 256)
        self.dec2 = DecoderBlock(256 + 128, 128)
        self.dec1 = DecoderBlock(128 + 64, 64)
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Attention modules
        self.attention1 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attention2 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attention3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsampling layers
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1_pool = self.pool(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3_pool)
        
        # Attention mechanisms
        att1 = self.attention1(enc1)
        att2 = self.attention2(enc2)
        att3 = self.attention3(enc3)
        
        # Apply attention
        enc1_att = enc1 * att1
        enc2_att = enc2 * att2
        enc3_att = enc3 * att3
        
        # Decoder with skip connections
        bottleneck_up = self.up(bottleneck)
        dec3_in = torch.cat((bottleneck_up, enc3_att), dim=1)
        dec3 = self.dec3(dec3_in)
        
        dec3_up = self.up(dec3)
        dec2_in = torch.cat((dec3_up, enc2_att), dim=1)
        dec2 = self.dec2(dec2_in)
        
        dec2_up = self.up(dec2)
        dec1_in = torch.cat((dec2_up, enc1_att), dim=1)
        dec1 = self.dec1(dec1_in)
        
        # Final layer
        out = self.final(dec1)
        
        return out

if __name__ == "__main__":
    # Simple test
    model = LowLightEnhancementNet()
    test_input = torch.randn(1, 3, 256, 256)
    output = model(test_input)
    print(f"Output shape: {output.shape}")

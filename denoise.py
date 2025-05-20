# denoising_model.py
import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        
        # First layer
        layers = [
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ]
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        noise = self.dncnn(x)
        # Residual learning: predict the noise, then subtract it from the input
        return torch.clamp(x - noise, 0, 1)

if __name__ == "__main__":
    # Simple test
    model = DnCNN()
    test_input = torch.randn(1, 3, 256, 256)
    output = model(test_input)
    print(f"Output shape: {output.shape}")

import torch
import torch.nn as nn
from torchvision import models

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MHABlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, resolution=8, dropout=0.5):
        super(MHABlock, self).__init__()
        self.num_heads = num_heads
        self.pos_embedding = nn.Parameter(torch.randn(1, resolution * resolution, in_channels))
        self.norm = nn.LayerNorm(in_channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )

        # zero init (preserve ResNet behavior at start)
        self.mha.out_proj.weight.data.zero_()
        self.mha.out_proj.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        
        # Apply LayerNorm before Attention (Pre-Norm)
        x_norm = self.norm(x_flat)
        
        # Add Positional Embedding
        seq_len = x_norm.shape[1]
        x_with_pos = x_norm + self.pos_embedding[:, :seq_len, :]

        # Self-Attention
        # Note: pass the normalized, positioned features to MHA
        attn_output, _ = self.mha(x_with_pos, x_with_pos, x_with_pos)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_out = attn_output.permute(0, 2, 1).view(b, c, h, w)
        
        return x + x_out

class AttentionPooling(nn.Module):
    def __init__(self, in_channels, attention_type, num_heads=8):
        super(AttentionPooling, self).__init__()
        
        if attention_type == "se":
            self.attention = SEBlock(in_channels)
        elif attention_type == "mha":
            self.attention = MHABlock(in_channels, num_heads=num_heads, resolution=8, dropout=0.5)
        else:
            self.attention = nn.Identity()
            
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.attention(x)
        x = self.pool(x)
        return x

def build_model(backbone="resnet", num_classes=3, pretrained=True, attention=None):
    if backbone == "resnet":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        
        if attention:
            model.avgpool = AttentionPooling(in_features, attention)
            
        model.fc = nn.Linear(in_features, num_classes)
        
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        
        if attention:
            model.avgpool = AttentionPooling(1280, attention, num_heads=8) 
            
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "mobilenet":

        model = models.mobilenet_v3_small(weights='DEFAULT')
        in_features = model.classifier[3].in_features
        
        # Replace the final linear layer
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif backbone == "swin":

        model = models.swin_t(weights='DEFAULT')
        in_features = model.head.in_features
        
        # Replace the final linear layer for 3 classes
        model.head = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError("Unsupported backbone")
        
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class CrossModalityAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels//8, 1)
        self.key = nn.Conv3d(channels, channels//8, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        
    def forward(self, pet_feat, ct_feat):
        # Compute cross-attention weights
        Q = self.query(pet_feat)
        K = self.key(ct_feat)
        V = self.value(ct_feat)
        
        attn = torch.softmax(Q.flatten(2) @ K.flatten(2).transpose(1,2), dim=-1)
        return (V.flatten(2) @ attn.transpose(1,2)).view_as(pet_feat)

class LesionAttention(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv3d(in_channels + 1, 1, kernel_size=1)  # +1 for mask channel
    
    def forward(self, x, mask):
        # Downsample mask
        mask_down = F.avg_pool3d(mask, kernel_size=self.scale_factor, 
                               stride=self.scale_factor)
        
        # Ensure spatial dimensions match
        if mask_down.shape[2:] != x.shape[2:]:
            mask_down = F.interpolate(mask_down, size=x.shape[2:], 
                                    mode='trilinear', align_corners=False)
        
        # Concatenate features and mask
        x_in = torch.cat([x, mask_down], dim=1)
        
        # Compute attention weights
        attn = torch.sigmoid(self.conv(x_in))
        return x * attn

class LungCancerSubtypeModel(nn.Module):
    def __init__(self, num_subtypes=4):
        super().__init__()
        
        # PET stream
        self.pet_conv1 = ConvBlock(1, 32)
        self.pet_conv2 = ConvBlock(32, 64)
        self.pet_attn = LesionAttention(64, scale_factor=(4,4,4))
        self.pet_conv3 = ConvBlock(64, 128)
        
        # CT stream
        self.ct_conv1 = ConvBlock(1, 32)
        self.ct_conv2 = ConvBlock(32, 64)
        self.ct_attn = LesionAttention(64, scale_factor=(4,4,4))
        self.ct_conv3 = ConvBlock(64, 128)
        
        # Cross-modality attention
        self.pet_to_ct_attn = CrossModalityAttention(128)
        self.ct_to_pet_attn = CrossModalityAttention(128)
        
        # Fusion and classification
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_subtypes)
        )

    def forward(self, pet, ct, mask):
        # ------------------------------
        # PET Stream
        # ------------------------------
        x_pet = self.pet_conv1(pet)        # [B,32,84,84,100]
        x_pet = self.pet_conv2(x_pet)      # [B,64,42,42,50]
        x_pet = self.pet_attn(x_pet, mask) # Lesion-guided attention
        x_pet = self.pet_conv3(x_pet)      # [B,128,21,21,25]
        
        # ------------------------------
        # CT Stream
        # ------------------------------
        x_ct = self.ct_conv1(ct)           # [B,32,84,84,100]
        x_ct = self.ct_conv2(x_ct)         # [B,64,42,42,50]
        x_ct = self.ct_attn(x_ct, mask)    # Lesion-guided attention
        x_ct = self.ct_conv3(x_ct)         # [B,128,21,21,25]
        
        # ------------------------------
        # Cross-Modality Attention
        # ------------------------------
        # PET attends to CT features
        x_pet_attn = self.pet_to_ct_attn(x_pet, x_ct)
        
        # CT attends to PET features
        x_ct_attn = self.ct_to_pet_attn(x_ct, x_pet)
        
        # ------------------------------
        # Feature Fusion
        # ------------------------------
        fused = torch.cat([x_pet_attn, x_ct_attn], dim=1)  # [B,256,21,21,25]
        
        # ------------------------------
        # Classification Head
        # ------------------------------
        x = self.global_pool(fused)        # [B,256,1,1,1]
        x = x.view(x.size(0), -1)          # [B,256]
        return self.fc(x)                  # [B,num_subtypes]

if __name__ == "__main__":
    input_shape = (2, 128, 128, 176)  # (channels, height, width)
    model = DualBranchUNet3D(input_shape=input_shape)
    
    # Move model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Print summary (input size must be in (channels, H, W) format)
    summary(model, input_size=input_shape)
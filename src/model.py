import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class LesionAttention2D(nn.Module):
    """
    2D病灶注意力模块。
    在网络的浅层，利用病灶掩码（Mask）对特征图进行空间加权，
    引导网络更关注病灶相关区域。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 使用一个1x1卷积将特征图和下采样后的Mask融合，生成单通道的注意力图
        self.conv = nn.Conv2d(in_channels + 1, 1, kernel_size=1)

    def forward(self, feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map (torch.Tensor): 卷积网络提取的特征图, [B, C, H, W]。
            mask (torch.Tensor): 原始的病灶掩码, [B, 1, H_orig, W_orig]。

        Returns:
            torch.Tensor: 经过注意力加权后的特征图, [B, C, H, W]。
        """
        # 将mask下采样到与特征图相同的空间尺寸，使用双线性插值
        mask_downsampled = F.interpolate(mask, size=feature_map.shape[2:], mode='bilinear', align_corners=False)
        
        # 将特征图和下采样后的mask在通道维度上拼接
        x_in = torch.cat([feature_map, mask_downsampled], dim=1)
        
        # 计算注意力权重图，并用sigmoid归一化到0-1之间
        attention_weights = torch.sigmoid(self.conv(x_in))
        
        # 将注意力权重广播并应用到原始特征图上
        return feature_map * attention_weights

class CrossModalityAttention2D(nn.Module):
    """
    2D跨模态注意力模块。
    在网络的深层，实现不同模态特征之间的信息交互。
    """
    def __init__(self, channels: int):
        super().__init__()
        # 定义Query, Key, Value的线性变换
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_feat: torch.Tensor, key_value_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feat (torch.Tensor): 查询模态的特征图 (e.g., PET), [B, C, H, W]。
            key_value_feat (torch.Tensor): 提供键/值信息的模态特征图 (e.g., CT), [B, C, H, W]。

        Returns:
            torch.Tensor: 经过交叉注意力增强后的查询特征图, [B, C, H, W]。
        """
        batch_size, _, height, width = query_feat.size()
        
        # 计算 Q, K, V
        proj_q = self.query_conv(query_feat).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_k = self.key_conv(key_value_feat).view(batch_size, -1, height * width)
        proj_v = self.value_conv(key_value_feat).view(batch_size, -1, height * width)
        
        # 计算注意力分数
        energy = torch.bmm(proj_q, proj_k)
        attention = self.softmax(energy)
        
        # 应用注意力并重塑为原始特征图形状
        out = torch.bmm(proj_v, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)
        
        return out

class HierarchicalFusionResNet(nn.Module):
    """
    基于ResNet-18的分层融合肺癌亚型分类模型。
    - 在Layer1后进行病灶空间注意力引导。
    - 在Layer3后进行跨模态语义特征融合。
    - 使用共享的Layer4进行最终特征提炼。
    """
    def __init__(self, num_subtypes: int = 3, pretrained: bool = True):
        super().__init__()
        
        # --- 1. 加载两个独立的预训练ResNet-18 ---
        resnet_pet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        resnet_ct = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # --- 2. PET流的编码器部分 ---
        self.pet_stem = nn.Sequential(resnet_pet.conv1, resnet_pet.bn1, resnet_pet.relu, resnet_pet.maxpool)
        self.pet_layer1 = resnet_pet.layer1 # 输出: 64通道
        self.pet_layer2 = resnet_pet.layer2 # 输出: 128通道
        self.pet_layer3 = resnet_pet.layer3 # 输出: 256通道

        # --- 3. CT流的编码器部分 ---
        self.ct_stem = nn.Sequential(resnet_ct.conv1, resnet_ct.bn1, resnet_ct.relu, resnet_ct.maxpool)
        self.ct_layer1 = resnet_ct.layer1   # 输出: 64通道
        self.ct_layer2 = resnet_ct.layer2   # 输出: 128通道
        self.ct_layer3 = resnet_ct.layer3   # 输出: 256通道
        
        # --- 4. 注意力模块 ---
        # 早期病灶注意力 (作用于Layer1的输出，64通道)
        self.pet_lesion_attn = LesionAttention2D(in_channels=64)
        self.ct_lesion_attn = LesionAttention2D(in_channels=64)

        # 中期跨模态注意力 (作用于Layer3的输出，256通道)
        self.cross_attn_p2c = CrossModalityAttention2D(channels=256) # PET查询CT
        self.cross_attn_c2p = CrossModalityAttention2D(channels=256) # CT查询PET

        # --- 5. 融合与共享编码器 ---
        # 适配器卷积：将拼接后的512通道特征图降维回256，以匹配预训练Layer4的输入
        self.adapter_conv = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        # 共享的深层编码器
        self.shared_layer4 = resnet_pet.layer4 # 输出: 512通道

        # --- 6. 分类头 ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_subtypes)
        )

    def forward(self, pet: torch.Tensor, ct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # --- 浅层特征提取 ---
        pet_stem_out = self.pet_stem(pet)       # (B, 64, 56, 56)
        pet_l1_out = self.pet_layer1(pet_stem_out)
        
        ct_stem_out = self.ct_stem(ct)         # (B, 64, 56, 56)
        ct_l1_out = self.ct_layer1(ct_stem_out)

        # --- 早期空间引导 (Lesion Attention) ---
        pet_l1_attn = self.pet_lesion_attn(pet_l1_out, mask)
        ct_l1_attn = self.ct_lesion_attn(ct_l1_out, mask)
        
        # --- 中层特征提取 ---
        pet_l3_out = self.pet_layer3(self.pet_layer2(pet_l1_attn)) # (B, 256, 14, 14)
        ct_l3_out = self.ct_layer3(self.ct_layer2(ct_l1_attn))   # (B, 256, 14, 14)
        
        # --- 中期语义融合 (Cross-Modality Attention with Residual Connection) ---
        pet_enhanced = self.cross_attn_p2c(pet_l3_out, ct_l3_out)
        pet_fused = pet_l3_out + pet_enhanced
        
        ct_enhanced = self.cross_attn_c2p(ct_l3_out, pet_l3_out)
        ct_fused = ct_l3_out + ct_enhanced
        
        # --- 特征拼接与适配 ---
        fused = torch.cat([pet_fused, ct_fused], dim=1) # (B, 512, 14, 14)
        adapted = self.adapter_conv(fused)             # (B, 256, 14, 14)

        # --- 共享深层编码 ---
        final_features = self.shared_layer4(adapted)   # (B, 512, 7, 7)
        
        # --- 分类 ---
        x = self.global_pool(final_features)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        
        return output

# ===================================================================
# 测试代码
# ===================================================================
if __name__ == "__main__":
    # --- 配置 ---
    batch_size = 4
    num_classes = 3 # ADC, SCC, SCLC
    
    # --- 模型实例化 ---
    # 在实际训练中， pretrained 应设为 True
    model = HierarchicalFusionResNet(num_subtypes=num_classes, pretrained=False)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
    
    # --- 模拟输入 ---
    dummy_pet = torch.randn(batch_size, 3, 224, 224)
    dummy_ct = torch.randn(batch_size, 3, 224, 224)
    dummy_mask = torch.rand(batch_size, 1, 224, 224) # Mask值在0-1之间
    
    # --- 前向传播测试 ---
    print("\n--- Testing Forward Pass ---")
    print(f"Input PET shape:  {dummy_pet.shape}")
    print(f"Input CT shape:   {dummy_ct.shape}")
    print(f"Input Mask shape: {dummy_mask.shape}")
    
    try:
        output = model(dummy_pet, dummy_ct, dummy_mask)
        print(f"Output shape:     {output.shape}")
        
        # 验证输出维度是否正确
        assert output.shape == (batch_size, num_classes)
        print("\nModel forward pass test successful!")
    except Exception as e:
        print(f"\nModel forward pass failed: {e}")
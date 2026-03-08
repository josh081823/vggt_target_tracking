import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os
from PIL import Image
import torchvision.transforms.functional as TF
import re


def get_gt_mask(ref_image_tensor, patch_size):
    # 检测纯白背景 (RGB > 0.95, 对应像素值约 242)
    is_white = (ref_image_tensor > 0.95).all(dim=1, keepdim=True).to(ref_image_tensor.dtype)
    _, _, H, W = ref_image_tensor.shape
    # 下采样到 Patch 网格大小 [1, 1, PATCH_H, PATCH_W]
    # 使用 area 插值计算每个 patch 中白色像素的比例
    is_white_patch = torch.nn.functional.interpolate(
        is_white, size=(H // patch_size, W // patch_size), mode='area'
    )
    
    # 如果 patch 中绝大部分是白色 (>0.99)，则认为是背景，不参与匹配
    # TODO: 这个阈值可以调整，或者改为动态计算 (例如使用分位数)
    patch_mask = (is_white_patch < 0.99).view(-1)
    return patch_mask

def get_similarity(q_tokens, ref_tokens, valid_patch_mask=None, top_k=5, temp=0.05):
    """
    计算查询 token 和参考 token 之间的相似度分数。
    q_tokens: [P, C] - 当前帧的 patch token
    ref_tokens: [P, C] - 参考帧的 patch token
    valid_patch_mask: [P] - 参考帧中有效 patch 的掩码
    top_k: int - 考虑 top-k 个最相似的参考 patch
    temp: float - softmax 的温度系数
    """
    # 计算余弦相似度: [P, P]
    sim = torch.matmul(q_tokens, ref_tokens.T)
    
    # 如果有掩码，将无效参考 patch 的相似度设为负无穷
    if valid_patch_mask is not None:
        sim[:, ~valid_patch_mask.bool()] = -float('inf')
        
    # 选取 top-k 相似度及其索引
    # sim_topk: [P, top_k], idx_topk: [P, top_k]
    sim_topk, _ = torch.topk(sim, k=top_k, dim=-1)
    
    # 应用 softmax 获取权重
    weights = torch.nn.functional.softmax(sim_topk / temp, dim=-1)
    
    # 加权求和得到最终分数
    score = (weights * sim_topk).sum(dim=-1)
    
    return score

def refine_scores_with_self_attention(scores, tokens, quantile_threshold=0.8, temp=0.1):
    """
    使用帧内自注意力机制来优化分数图，以增强掩码的完整性。

    Args:
        scores (torch.Tensor): 初始分数图，形状为 [S, P]。
        tokens (torch.Tensor): 对应的归一化特征，形状为 [S, P, C]。
        quantile_threshold (float): 用于选择候选 patch 的分位数阈值。
        temp (float): self-attention 中 softmax 的温度系数。

    Returns:
        torch.Tensor: 经过优化的分数图，形状为 [S, P]。
    """
    S, P, C = tokens.shape
    # Ensure scores matches tokens dtype to avoid mismatch during assignment
    scores = scores.to(tokens.dtype)
    refined_scores = scores.clone()
    
    print("Refining scores with intra-frame self-attention...")

    for i in range(S):
        scores_i = scores[i]
        tokens_i = tokens[i]
        
        # 1. 基于初始分数选择候选 patch
        # 使用分位数阈值来动态确定哪些 patch 参与注意力计算
        valid_scores_i = scores_i[torch.isfinite(scores_i)]
        if valid_scores_i.numel() == 0:
            continue
        
        threshold = torch.quantile(valid_scores_i, quantile_threshold)
        candidate_indices = (scores_i > threshold).nonzero(as_tuple=False).squeeze(-1)
        
        # 如果候选 patch 太少，则跳过
        if candidate_indices.numel() < 2:
            continue
            
        candidate_features = tokens_i[candidate_indices]  # [k, C]
        candidate_scores = scores_i[candidate_indices]    # [k]
        
        # 2. 计算 self-attention 权重
        # 特征已经 L2 归一化，所以 matmul 结果是余弦相似度
        attn_logits = torch.matmul(candidate_features, candidate_features.T) / (C**0.5)
        attn_weights = torch.nn.functional.softmax(attn_logits / temp, dim=-1)  # [k, k]
        
        # 3. 分数传播：每个候选 patch 的新分数是其邻居（包括自身）分数的加权平均
        new_candidate_scores = torch.matmul(attn_weights, candidate_scores).to(tokens.dtype)  # [k]
        
        # 4. 更新总分数图 (平滑融合新旧分数)
        # refined_scores[i, candidate_indices] = (scores_i[candidate_indices] + new_candidate_scores) / 2.0
        refined_scores[i, candidate_indices] = new_candidate_scores

        
    return refined_scores

def generate_ref_mask(frame_features_list, layers_to_use=[2, 3], threshold_quantile=0.7, ref_patch_mask=None):
    """
    方法说明：
    使用 VGGT 交替注意力的指定层共同生成目标物体掩码。
    1. 计算指定 VGGT 层的 Frame 特征与参考帧的相似度。
    2. 将所有相似度分数平均融合，得到最终的置信度图。
    3. (新) 对每一帧的候选 Patch 应用自注意力机制，利用帧内一致性平滑和完善分数。
       使用所有选中层的 Frame token 拼接后的特征进行自注意力计算。
    4. 使用动态阈值生成二值掩码。
    """
    print(f"正在生成组合伪标签 (VGGT Layers {layers_to_use})...")
    
    # 获取维度信息
    sample_token = frame_features_list[layers_to_use[0]]
    B, S, P, C = sample_token.shape
    device = sample_token.device
    dtype = sample_token.dtype

    # 2. VGGT Layers Score
    layer_scores_sum = torch.zeros((B, S, P), device=device, dtype=dtype)
    concat_features_list = []
    
    for l_idx in layers_to_use:
        # frame_features_list[l_idx]: [B, S, P, C]
        frame_tokens = frame_features_list[l_idx]
        
        frame_norm = torch.nn.functional.normalize(frame_tokens, p=2, dim=-1)
        
        ref_frame = frame_norm[:, -1] # [B, P, C]
        
        # 收集特征以备后续 self-attention 使用
        concat_features_list.append(frame_norm)
        
        for b in range(B):
            ref_f = ref_frame[b]
            mask_b = None
            if ref_patch_mask is not None:
                if ref_patch_mask.numel() == B * P:
                    mask_b = ref_patch_mask.view(B, P)[b]
                elif ref_patch_mask.numel() == P:
                    mask_b = ref_patch_mask

            for i in range(S):
                s_frame = get_similarity(frame_norm[b, i], ref_f, mask_b, top_k=5, temp=0.2)
                layer_scores_sum[b, i] += s_frame
            
    # 3. Combine
    final_scores = layer_scores_sum / len(layers_to_use)

    # 4. (新) 使用 Self-Attention 优化分数
    # 将所有层的 frame token 在通道维度拼接
    concat_features = torch.cat(concat_features_list, dim=-1) # [B, S, P, len * C]
    concat_features = torch.nn.functional.normalize(concat_features, p=2, dim=-1) # 重新归一化
    
    for b in range(B):
        final_scores[b] = refine_scores_with_self_attention(final_scores[b], concat_features[b], quantile_threshold=0.92, temp=0.1)

    # 5. Threshold
    # Normalize final_scores to [0, 1] per frame
    mask = torch.zeros_like(final_scores)
    for b in range(B):
        for i in range(S):
            scores = final_scores[b, i]
            finite_mask = torch.isfinite(scores)
            if finite_mask.any():
                min_val = scores[finite_mask].min()
                max_val = scores[finite_mask].max()
                if max_val > min_val:
                    final_scores[b, i] = (scores - min_val) / (max_val - min_val)

        valid_scores = final_scores[b][torch.isfinite(final_scores[b])]
        threshold = torch.quantile(valid_scores, threshold_quantile) if valid_scores.numel() > 0 else 0.7
        print(f"Batch {b}: 使用动态阈值 (q={threshold_quantile}): {threshold:.4f}")
        mask[b] = (final_scores[b] > threshold).to(dtype)
    
    return mask, final_scores

def visualize_mask_and_scores(masks, scores, patch_h, patch_w, save_path):
    """
    Visualizes the generated masks and their corresponding scores for debugging.

    Args:
        masks (torch.Tensor): The binary mask tensor, shape [B, S, P].
        scores (torch.Tensor): The score tensor, shape [B, S, P].
        patch_h (int): The height of the patch grid.
        patch_w (int): The width of the patch grid.
        save_path (str or Path): The path to save the visualization.
    """
    # Assuming B=1 for visualization, we take the first item in the batch.
    if masks.shape[0] > 1 or scores.shape[0] > 1:
        print(f"Info: Visualizing only the first item of a batch with size {masks.shape[0]}.")
    
    masks_vis = masks[0].cpu()   # Shape [S, P]
    scores_vis = scores[0].cpu() # Shape [S, P]
    
    S = masks_vis.shape[0]
    print(f"Visualizing masks for {S} frames...")
    
    fig, axes = plt.subplots(S, 2, figsize=(10, 3 * S), squeeze=False)

    for i in range(S):
        score_map = scores_vis[i].reshape(patch_h, patch_w).numpy()
        mask_map = masks_vis[i].reshape(patch_h, patch_w).numpy()
        
        # Plot Score Map
        im = axes[i, 0].imshow(score_map, cmap='viridis', vmin=0, vmax=1)
        axes[i, 0].set_title(f"Frame {i} Similarity (Max: {score_map.max():.2f})")
        fig.colorbar(im, ax=axes[i, 0])
        axes[i, 0].axis('off')

        # Plot Mask Map
        axes[i, 1].imshow(mask_map, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Frame {i} Mask")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved debug visualization to: {save_path}")

def visualize_camera_attention(attn_map, patch_h, patch_w, save_path, layer_idx, patch_start_idx=5):
    """
    Visualizes the attention map of the camera token towards patch tokens.
    attn_map: [B*S, num_heads, N, N]
    """
    # Assuming B=1, iterate over S frames
    S = attn_map.shape[0]
    
    fig, axes = plt.subplots(1, S, figsize=(5 * S, 5))
    if S == 1: axes = [axes]
    
    for i in range(S):
        # Average over heads: [N, N]
        attn_avg = attn_map[i].mean(dim=0)
        
        # Camera token is typically at index 0
        # We want to see its attention towards Patch Tokens (index patch_start_idx onwards)
        camera_attn = attn_avg[0, patch_start_idx:] # [P_patches]
        
        if camera_attn.numel() != patch_h * patch_w:
            print(f"Warning: Attention map size {camera_attn.numel()} does not match patch grid {patch_h}x{patch_w}")
            continue
            
        attn_img = camera_attn.reshape(patch_h, patch_w).numpy()
        
        ax = axes[i]
        im = ax.imshow(attn_img, cmap='viridis')
        ax.set_title(f"L{layer_idx} Frame {i} Cam Attn")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved camera attention visualization to: {save_path}")
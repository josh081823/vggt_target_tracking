# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.utils.target_mask import get_gt_mask, generate_ref_mask, visualize_camera_attention

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self, images: torch.Tensor, early_stage_masking: bool = False
    ) -> Tuple[List[torch.Tensor], int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            early_stage_masking (bool): Whether to apply early stage masking to layers 1-5.

        Returns:
            (list[torch.Tensor], int, Optional[torch.Tensor], Optional[torch.Tensor]):
                - The list of outputs from the attention blocks,
                - The patch_start_idx indicating where patch tokens begin.
                - The generated dynamic mask if early_stage_masking is True.
                - The generated scores if early_stage_masking is True.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        dynamic_mask = None
        scores = None
        if not self.training and early_stage_masking:
            valid_patch_mask = get_gt_mask(images[:, -1], self.patch_size)


        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # Prepare attention masks for early-stage masking (layers 1-5)
        frame_attn_mask = None
        global_attn_mask = None

        if not self.training and early_stage_masking:
            with torch.no_grad():
                temp_tokens = tokens.clone()
                temp_frame_idx = 0
                temp_global_idx = 0
                layers_to_use = [1, 2]
                max_layer = 2
                frame_features_list = []
                current_layer = 0

                for _ in range(max_layer+1):
                    for attn_type in self.aa_order:
                        if attn_type == "frame":
                            temp_tokens, temp_frame_idx, frame_intermediates = self._process_frame_attention(
                                temp_tokens, B, S, P, C, temp_frame_idx, pos=pos, attn_mask=None
                            )
                        elif attn_type == "global":
                            temp_tokens, temp_global_idx, global_intermediates = self._process_global_attention(
                                temp_tokens, B, S, P, C, temp_global_idx, pos=pos, attn_mask=None
                            )
                        else:
                            raise ValueError(f"Unknown attention type: {attn_type}")

                    for i in range(len(frame_intermediates)):
                        frame_features_list.append(frame_intermediates[i][:, :, self.patch_start_idx:, :])

                    current_layer += len(frame_intermediates)

                print("Generating reference mask...")
                dynamic_mask, scores = generate_ref_mask(frame_features_list, layers_to_use=layers_to_use, threshold_quantile=0.887, ref_patch_mask=valid_patch_mask)
                
                # Cleanup
                del temp_tokens, frame_intermediates, global_intermediates, frame_features_list
                torch.cuda.empty_cache()

            if dynamic_mask is not None:
                # Flatten dynamic_mask if it's 4D [B, S, H, W] -> [B, S, P_patches]
                if dynamic_mask.dim() == 4:
                    dynamic_mask = dynamic_mask.flatten(2)

                # Pad mask for special tokens (camera + registers) which are always kept (False -> Keep)
                # dynamic_mask: [B, S, P_patches] -> full_mask: [B, S, P]
                # P = patch_start_idx + P_patches
                
                # Attention mask logic: True means suppress (ignore), False means attend (keep).
                # dynamic_mask is 1 (True) for target, 0 (False) for background.
                # We want to keep Target and Special tokens, and suppress Background.
                mask_pad = torch.zeros((B, S, self.patch_start_idx), dtype=torch.bool, device=dynamic_mask.device)
                # Invert dynamic_mask: 1 (Target) -> False (Keep), 0 (Background) -> True (Suppress)
                patches_mask = ~dynamic_mask.bool()
                full_mask = torch.cat([mask_pad, patches_mask], dim=2)

                # Frame attention mask: [B*S, 1, 1, P]
                # We want to mask positions where full_mask is True.
                frame_attn_mask = full_mask.view(B * S, P).unsqueeze(1).unsqueeze(1)

                # Global attention mask: [B, 1, 1, S*P]
                # Flatten S and P dimensions
                global_attn_mask = full_mask.view(B, S * P).unsqueeze(1).unsqueeze(1)
                print("Early stage attention masks generated.")

        frame_idx = 0
        global_idx = 0
        output_list = []

        vis_cam_attention = False
        # Set to True to enable camera attention visualization
        if vis_cam_attention:
            # Enable attention saving for the last layer to visualize camera attention
            if len(self.frame_blocks) > 0:
                self.frame_blocks[-1].attn.save_attention = True
            if len(self.global_blocks) > 0:
                self.global_blocks[-1].attn.save_attention = True

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, attn_mask=frame_attn_mask
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, attn_mask=global_attn_mask
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        # Visualize camera attention if available
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        for i, blk in enumerate(self.frame_blocks):
            if blk.attn.attn_map is not None:
                visualize_camera_attention(blk.attn.attn_map, patch_h, patch_w, f"camera_attn_frame_layer_{i}.png", i, self.patch_start_idx)
                blk.attn.attn_map = None
                blk.attn.save_attention = False

        for i, blk in enumerate(self.global_blocks):
            if blk.attn.attn_map is not None:
                # attn_map: [B, H, S*P, S*P] -> [S, H, P, P] (intra-frame)
                # Assuming B=1
                attn_map = blk.attn.attn_map
                if attn_map.shape[0] == 1:
                    # Reshape to extract intra-frame attention
                    # [1, H, S*P, S*P] -> [1, H, S, P, S, P]
                    attn_map = attn_map.view(1, -1, S, P, S, P)
                    # Extract diagonal on S dimensions (dim 2 and 4)
                    # Result: [1, H, P, P, S]
                    attn_map = attn_map.diagonal(dim1=2, dim2=4)
                    # Permute to [S, 1, H, P, P] -> [S, H, P, P]
                    attn_map = attn_map.permute(4, 0, 1, 2, 3).squeeze(1)
                    
                    visualize_camera_attention(attn_map, patch_h, patch_w, f"camera_attn_global_layer_{i}.png", i, self.patch_start_idx)
                
                blk.attn.attn_map = None
                blk.attn.save_attention = False

        del concat_inter
        del frame_intermediates
        del global_intermediates

        # Reshape mask and scores for easier visualization
        if dynamic_mask is not None:
            dynamic_mask = dynamic_mask.view(B, S, patch_h, patch_w)
        if scores is not None:
            scores = scores.view(B, S, patch_h, patch_w)

        return output_list, self.patch_start_idx, dynamic_mask, scores

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, attn_mask=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            # Apply mask only to layers 1-5 (indices 0-4)
            mask = attn_mask if (frame_idx < 5 and attn_mask is not None) else None

            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, mask, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos, attn_mask=mask)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, attn_mask=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            # Apply mask only to layers 1-5 (indices 0-4)
            mask = attn_mask if (global_idx < 5 and attn_mask is not None) else None

            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, mask, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=mask)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

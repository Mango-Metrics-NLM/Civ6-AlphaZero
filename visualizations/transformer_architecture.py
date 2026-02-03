"""
Create a matplotlib visualization of the Civ6 Transformer Network architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 10)
ax.set_ylim(0, 30)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E8F4F8',
    'embedding': '#B8E6F0',
    'transformer': '#66C2DD',
    'attention': '#4AA5C0',
    'ffn': '#2E8BA0',
    'head': '#F4A261',
    'output': '#E76F51',
    'norm': '#90E0EF',
    'rope': '#CAF0F8',
    'moe': '#A8DADC'
}

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False, alpha=0.8):
    """Draw a fancy box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor='#264653',
        facecolor=color,
        alpha=alpha
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize,
        weight=weight,
        color='#264653'
    )

def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=20,
        linewidth=2,
        color='#264653',
        alpha=0.7
    )
    ax.add_patch(arrow)

# Title
ax.text(5, 29, 'Civ6 Transformer Network Architecture', 
        ha='center', va='top', fontsize=18, weight='bold', color='#264653')

# Current Y position for drawing (top to bottom)
y_pos = 27

# ============ INPUT LAYER ============
ax.text(5, y_pos, 'Input Layer', ha='center', va='center', 
        fontsize=14, weight='bold', color='#264653')
y_pos -= 1

# Spatial input
draw_box(ax, 0.5, y_pos-0.8, 3.5, 0.7, 
         'Spatial Input\n(B, 13, 16, 16)', colors['input'], 9)

# Vector input
draw_box(ax, 6, y_pos-0.8, 3.5, 0.7, 
         'Vector Input\n(B, 64)', colors['input'], 9)

y_pos -= 1.5

# ============ EMBEDDING LAYER ============
ax.text(5, y_pos, 'Embedding Layer', ha='center', va='center', 
        fontsize=14, weight='bold', color='#264653')
y_pos -= 1

# Patch embedding
draw_box(ax, 0.5, y_pos-1.2, 3.5, 1.0, 
         'Patch Embedding (ViT)\nConv2d(kernel=4, stride=4)\n→ (B, 16, 256)', 
         colors['embedding'], 8)
draw_arrow(ax, 2.25, y_pos+1.5, 2.25, y_pos+0.8)

# Vector projection
draw_box(ax, 6, y_pos-1.2, 3.5, 1.0, 
         'Vector Projection\nLinear(64 → 256)\n→ (B, 1, 256)', 
         colors['embedding'], 8)
draw_arrow(ax, 7.75, y_pos+1.5, 7.75, y_pos+0.8)

y_pos -= 1.8

# ============ TOKEN CONCATENATION ============
draw_box(ax, 2, y_pos-0.8, 6, 0.7, 
         'Token Concat: [CLS] + Vector + Patches\n→ (B, 18, 256)', 
         colors['embedding'], 9, bold=True)
draw_arrow(ax, 2.25, y_pos+1.0, 3.5, y_pos+0.5)
draw_arrow(ax, 7.75, y_pos+1.0, 6.5, y_pos+0.5)
y_pos -= 1.2

# Position embedding
draw_box(ax, 2, y_pos-0.8, 6, 0.7, 
         '+ Learned Position Embeddings\n→ (B, 18, 256)', 
         colors['norm'], 9)
draw_arrow(ax, 5, y_pos+0.6, 5, y_pos+0.2)
y_pos -= 1.5

# ============ TRANSFORMER BLOCKS ============
ax.text(5, y_pos, 'Transformer Encoder (8 Blocks)', ha='center', va='center', 
        fontsize=14, weight='bold', color='#264653')
y_pos -= 1

# Draw one detailed block
block_start_y = y_pos

# Block container
block_height = 8
draw_box(ax, 1, y_pos-block_height-0.2, 8, block_height+0.4, 
         '', colors['transformer'], alpha=0.2)

# Input
draw_arrow(ax, 5, y_pos+0.5, 5, y_pos-0.2)

# Pre-norm 1
y_pos -= 0.7
draw_box(ax, 2.5, y_pos-0.5, 5, 0.4, 
         'RMSNorm', colors['norm'], 8)
draw_arrow(ax, 5, y_pos+0.3, 5, y_pos-0.1)

# Multi-head attention
y_pos -= 1
draw_box(ax, 1.5, y_pos-1.8, 7, 1.6, '', colors['attention'], alpha=0.3)
ax.text(5, y_pos-0.1, 'Multi-Head Self-Attention', ha='center', va='center', 
        fontsize=10, weight='bold', color='#264653')

# QKV projection
y_pos -= 0.6
draw_box(ax, 2.5, y_pos-0.5, 5, 0.4, 
         'QKV Projection (Linear)\n→ (B, 18, 8, 32) each', colors['attention'], 7)
draw_arrow(ax, 5, y_pos+0.3, 5, y_pos-0.1)

# RoPE
y_pos -= 0.8
draw_box(ax, 1.8, y_pos-0.6, 2.8, 0.5, 
         'RoPE\n(Q, K rotation)', colors['rope'], 7)
draw_arrow(ax, 5, y_pos+0.6, 3.2, y_pos+0.1)

# Attention
draw_box(ax, 5.4, y_pos-0.6, 2.8, 0.5, 
         'Scaled Dot-Product\nAttention', colors['attention'], 7)
draw_arrow(ax, 3.2, y_pos-0.4, 5.4, y_pos-0.4)

y_pos -= 1
# Output projection
draw_box(ax, 2.5, y_pos-0.5, 5, 0.4, 
         'Output Projection (Linear)', colors['attention'], 7)
draw_arrow(ax, 6.8, y_pos+0.4, 5.5, y_pos+0.2)
draw_arrow(ax, 5, y_pos+0.8, 5, y_pos-0.1)

# Residual connection for attention
draw_arrow(ax, 0.8, block_start_y-0.7, 0.8, y_pos, style='-')
draw_arrow(ax, 0.8, y_pos, 2.5, y_pos, style='->')

y_pos -= 0.8
# Add
draw_box(ax, 4.5, y_pos-0.4, 1, 0.3, '+', colors['norm'], 10, bold=True)
draw_arrow(ax, 5, y_pos+0.3, 5, y_pos-0.1)

# Pre-norm 2
y_pos -= 0.7
draw_box(ax, 2.5, y_pos-0.5, 5, 0.4, 
         'RMSNorm', colors['norm'], 8)
draw_arrow(ax, 5, y_pos+0.3, 5, y_pos-0.1)

# Feed-forward / MoE
y_pos -= 1
draw_box(ax, 1.5, y_pos-1.5, 7, 1.3, '', colors['moe'], alpha=0.3)

# MoE or FFN choice
ax.text(5, y_pos-0.1, 'Feed-Forward / MoE (conditional)', ha='center', va='center', 
        fontsize=10, weight='bold', color='#264653')

y_pos -= 0.6
draw_box(ax, 2, y_pos-0.6, 2.5, 0.5, 
         'Router\n(if MoE)', colors['moe'], 7)
draw_box(ax, 5.5, y_pos-0.6, 2.5, 0.5, 
         'SwiGLU\n(w1(x) * w3(x))', colors['ffn'], 7)
draw_arrow(ax, 5, y_pos+0.4, 3.25, y_pos-0.1)
draw_arrow(ax, 5, y_pos+0.4, 6.75, y_pos-0.1)

y_pos -= 0.9
draw_box(ax, 2.5, y_pos-0.5, 5, 0.4, 
         'Output (Gated MLP)', colors['ffn'], 7)
draw_arrow(ax, 3.25, y_pos+0.3, 4.5, y_pos+0.2)
draw_arrow(ax, 6.75, y_pos+0.3, 5.5, y_pos+0.2)
draw_arrow(ax, 5, y_pos+0.5, 5, y_pos-0.1)

# Residual connection for FFN
residual_start = block_start_y - 5.5
draw_arrow(ax, 0.5, residual_start, 0.5, y_pos, style='-')
draw_arrow(ax, 0.5, y_pos, 2.5, y_pos, style='->')

y_pos -= 0.8
# Add
draw_box(ax, 4.5, y_pos-0.4, 1, 0.3, '+', colors['norm'], 10, bold=True)
draw_arrow(ax, 5, y_pos+0.3, 5, y_pos-0.1)

# Stochastic depth
y_pos -= 0.5
ax.text(5, y_pos, '↓ (Stochastic Depth / DropPath)', ha='center', va='center', 
        fontsize=8, style='italic', color='#264653')

y_pos -= 1

# Repeat indication
draw_box(ax, 2, y_pos-0.6, 6, 0.5, 
         '↻ Repeat 8x (Transformer Blocks)', colors['transformer'], 9, bold=True)
draw_arrow(ax, 5, y_pos+0.4, 5, y_pos-0.1)

y_pos -= 1.2

# ============ FINAL NORM ============
draw_box(ax, 2.5, y_pos-0.5, 5, 0.4, 
         'Final RMSNorm', colors['norm'], 9, bold=True)
draw_arrow(ax, 5, y_pos+0.5, 5, y_pos-0.1)

y_pos -= 1

# ============ POOLING ============
draw_box(ax, 2.5, y_pos-0.6, 5, 0.5, 
         'Pooling (CLS Token)\n→ (B, 256)', colors['transformer'], 9, bold=True)
draw_arrow(ax, 5, y_pos+0.5, 5, y_pos-0.1)

y_pos -= 1.2

# ============ OUTPUT HEADS ============
ax.text(5, y_pos, 'Output Heads', ha='center', va='center', 
        fontsize=14, weight='bold', color='#264653')
y_pos -= 0.8

# Split for two heads
draw_arrow(ax, 5, y_pos+0.5, 2.25, y_pos+0.1)
draw_arrow(ax, 5, y_pos+0.5, 7.75, y_pos+0.1)

# Policy head
draw_box(ax, 0.5, y_pos-2.5, 3.5, 2.3, '', colors['head'], alpha=0.3)
ax.text(2.25, y_pos-0.3, 'Policy Head', ha='center', va='center', 
        fontsize=11, weight='bold', color='#264653')
draw_box(ax, 1, y_pos-1, 2.5, 0.4, 'Linear(256→256)', colors['head'], 8)
draw_arrow(ax, 2.25, y_pos, 2.25, y_pos-0.6)
draw_box(ax, 1, y_pos-1.6, 2.5, 0.4, 'GELU + Dropout', colors['head'], 8)
draw_arrow(ax, 2.25, y_pos-1.1, 2.25, y_pos-1.2)
draw_box(ax, 1, y_pos-2.2, 2.5, 0.4, 'Linear(256→381)', colors['head'], 8)
draw_arrow(ax, 2.25, y_pos-1.7, 2.25, y_pos-1.8)

# Value head
draw_box(ax, 6, y_pos-2.5, 3.5, 2.3, '', colors['head'], alpha=0.3)
ax.text(7.75, y_pos-0.3, 'Value Head', ha='center', va='center', 
        fontsize=11, weight='bold', color='#264653')
draw_box(ax, 6.5, y_pos-1, 2.5, 0.4, 'Linear(256→128)', colors['head'], 8)
draw_arrow(ax, 7.75, y_pos, 7.75, y_pos-0.6)
draw_box(ax, 6.5, y_pos-1.6, 2.5, 0.4, 'GELU + Dropout', colors['head'], 8)
draw_arrow(ax, 7.75, y_pos-1.1, 7.75, y_pos-1.2)
draw_box(ax, 6.5, y_pos-2.2, 2.5, 0.4, 'Linear(128→1) + Tanh', colors['head'], 8)
draw_arrow(ax, 7.75, y_pos-1.7, 7.75, y_pos-1.8)

y_pos -= 3

# ============ OUTPUTS ============
draw_box(ax, 0.5, y_pos-0.7, 3.5, 0.6, 
         'Policy Logits\n(B, 381)', colors['output'], 9, bold=True)
draw_arrow(ax, 2.25, y_pos+0.5, 2.25, y_pos-0.1)

draw_box(ax, 6, y_pos-0.7, 3.5, 0.6, 
         'Value Estimate\n(B,)', colors['output'], 9, bold=True)
draw_arrow(ax, 7.75, y_pos+0.5, 7.75, y_pos-0.1)

# ============ LEGEND ============
legend_y = 1.5
legend_elements = [
    ('Input/Output', colors['input']),
    ('Embedding', colors['embedding']),
    ('Transformer', colors['transformer']),
    ('Attention', colors['attention']),
    ('FFN/MoE', colors['ffn']),
    ('Normalization', colors['norm']),
    ('Output Heads', colors['head']),
]

ax.text(5, legend_y+0.5, 'Legend', ha='center', va='center', 
        fontsize=12, weight='bold', color='#264653')

for i, (label, color) in enumerate(legend_elements):
    x = 1.5 + (i % 4) * 2
    y = legend_y - 0.5 - (i // 4) * 0.6
    
    box = FancyBboxPatch(
        (x, y), 0.3, 0.3,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor='#264653',
        facecolor=color,
        alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + 0.5, y + 0.15, label, ha='left', va='center', 
            fontsize=8, color='#264653')

# Add configuration info in bottom right
config_text = """Key Features:
• Vision Transformer (ViT) patch embedding
• 8 Transformer layers
• 8 attention heads (32 dim each)
• Rotary Position Embeddings (RoPE)
• Pre-LayerNorm architecture
• SwiGLU activation (gated FFN)
• Optional Mixture of Experts (MoE)
• Stochastic Depth regularization
• Dual heads: Policy (381) + Value (1)"""

ax.text(9.5, 10, config_text, ha='right', va='top', 
        fontsize=7, family='monospace', color='#264653',
        bbox=dict(boxstyle='round', facecolor='#F1FAEE', alpha=0.8, edgecolor='#264653'))

plt.tight_layout()
plt.savefig('transformer_network_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("[SUCCESS] Transformer network architecture diagram saved to: transformer_network_architecture.png")

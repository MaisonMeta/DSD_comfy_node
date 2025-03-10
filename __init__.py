"""
Diffusion Self-Distillation (DSD) ComfyUI Extension
A ComfyUI extension for running the Flux Diffusion Self-Distillation process.
"""

# Import the main node implementation directly 
from .dsd_comfy_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Print a debug message to verify the initialization
print("=" * 60)
print("DSD ComfyUI Extension initialized. Available nodes:")
for node_key in NODE_CLASS_MAPPINGS.keys():
    print(f"  - {node_key}: {NODE_DISPLAY_NAME_MAPPINGS.get(node_key, '')}")
print("=" * 60)

# Export the variables
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

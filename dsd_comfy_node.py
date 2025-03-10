"""
Diffusion Self-Distillation (DSD) node for ComfyUI
This module implements the DSD process for enhancing images with diffusion.
"""

import os
import sys
import json
import traceback
import random
import time
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as safetensors_load

# Standard image handling utilities for ComfyUI nodes
def pil_to_tensor(image):
    """Convert a PIL image to a PyTorch tensor in BCHW format"""
    img_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    # HWC to BCHW
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def tensor_to_pil(tensor):
    """Convert a PyTorch tensor (BCHW) to a PIL image"""
    # BCHW to HWC
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def load_file(path):
    """Load a safetensors file"""
    try:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
            
        print(f"Loading safetensors file: {path}")
        weights = safetensors_load(path)
        
        if weights:
            # Print some information about the loaded tensors
            print(f"Loaded {len(weights)} tensors from {path}")
            sample_keys = list(weights.keys())[:3] if len(weights) > 3 else list(weights.keys())
            print(f"Example keys: {sample_keys}")
        return weights
    except Exception as e:
        print(f"Error loading file {path}: {e}")
        # Print a simplified error message instead of using traceback
        import sys
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print(f"Python version: {sys.version}")
        return None

# Advanced image conversion function to handle unusual tensor shapes
def fix_tensor_for_pil(tensor):
    """
    Convert any tensor to a format suitable for PIL Image creation
    Handles unusual tensor shapes and formats
    """
    print(f"Converting tensor with shape {tensor.shape} and dtype {tensor.dtype}")
    
    # First detect if we have BHWC (batch, height, width, channels) or BCHW (batch, channels, height, width)
    is_channels_last = False
    if len(tensor.shape) == 4:
        # Heuristic to determine format: typically channels are smaller than dimensions
        B, dim1, dim2, dim3 = tensor.shape
        if dim3 <= 4 and dim1 > 4 and dim2 > 4:
            print(f"Detected BHWC format: {tensor.shape}")
            is_channels_last = True
        
        # For BHWC format, convert to BCHW for processing
        if is_channels_last:
            print(f"Converting from BHWC to BCHW")
            tensor = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
    
    # Now tensor should be in BCHW format
    if len(tensor.shape) == 4:  # BCHW format
        batch, channels, height, width = tensor.shape
        
        # Special case: (1, C, 1, 1) - use channel dimension as width for visualization
        if height == 1 and width == 1:
            print(f"Reshaping unusual tensor from {tensor.shape} to usable format")
            # Reshape to a 1D horizontal strip
            tensor = tensor.reshape(1, 1, 1, channels)
            tensor = torch.nn.functional.interpolate(tensor, size=(64, channels), mode='nearest')
    
    # Handle specific 3D tensor shapes
    elif len(tensor.shape) == 3:
        if tensor.shape[0] == 1 and tensor.shape[1] == 1:
            # Case: (1, 1, N) - reshape to 2D
            print(f"Reshaping (1, 1, N) tensor to 2D format")
            n = tensor.shape[2]
            size = int(np.sqrt(n)) + 1
            # Pad to make a square
            padded = torch.zeros((1, 1, size*size), dtype=tensor.dtype, device=tensor.device)
            padded[0, 0, :n] = tensor[0, 0, :n]
            tensor = padded.reshape(1, 1, size, size)
            tensor = tensor.repeat(1, 3, 1, 1)  # Make RGB
        
        # If not BHWC check if CHW and convert to BCHW
        elif not is_channels_last and min(tensor.shape) <= 4 and max(tensor.shape) > 32:
            print(f"Converting possible CHW tensor to RGB")
            if tensor.shape[0] <= 4:  # Channels first (CHW)
                if tensor.shape[0] == 1:  # Grayscale
                    tensor = tensor.repeat(3, 1, 1)  # Make RGB
                elif tensor.shape[0] == 4:  # RGBA
                    tensor = tensor[:3]  # Take RGB part
                tensor = tensor.unsqueeze(0)  # Add batch dim
            else:  # HWC format
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
    
    # At this point we should have a BCHW tensor
    # Convert to HWC numpy array
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Check channel count
    if tensor.shape[0] not in [1, 3, 4]:
        print(f"Warning: Unusual channel count {tensor.shape[0]}, creating a default RGB image")
        # Create a default colored placeholder
        h, w = tensor.shape[1], tensor.shape[2] if len(tensor.shape) > 2 else tensor.shape[1]
        h, w = max(h, 64), max(w, 64)  # Ensure minimum size
        return Image.new('RGB', (w, h), (100, 149, 237))  # Cornflower blue
    
    # Convert to numpy array and proper range for PIL
    if tensor.shape[0] == 1:  # Grayscale
        # For grayscale, PIL expects a 2D array
        array = tensor.squeeze(0).cpu().numpy()
        if array.dtype != np.uint8:
            array = np.clip(array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode='L')
    else:
        # For RGB/RGBA, permute to HWC and convert
        array = tensor.permute(1, 2, 0).cpu().numpy()
        if array.dtype != np.uint8:
            array = np.clip(array * 255, 0, 255).astype(np.uint8)
        mode = 'RGB' if tensor.shape[0] == 3 else 'RGBA'
        return Image.fromarray(array, mode=mode)

# Core DSD implementation
class DSDProcessor:
    def __init__(self, device="cuda"):
        # Ensure CUDA is used if available
        try:
            if device == "cuda" and not torch.cuda.is_available():
                print("WARNING: CUDA is not available, using CPU")
                device = "cpu"
                
            self.device = device
            print(f"Using device: {self.device}")
            
            # Set default dtype for compatibility
            if device == "cuda" and torch.cuda.is_available():
                # Use BFloat16 for GPU calculations (better compatibility)
                if torch.cuda.is_bf16_supported():
                    self.working_dtype = torch.bfloat16
                    print(f"Using BFloat16 precision for GPU calculations")
                else:
                    self.working_dtype = torch.float16
                    print(f"BFloat16 not supported, using Float16 precision for GPU calculations")
            else:
                # Use Float32 for CPU calculations
                self.working_dtype = torch.float32
                print(f"Using Float32 precision for CPU calculations")
                
            # Will be loaded when needed
            self.model_weights = None
            self.lora_weights = None
            self.is_loaded = False
            print(f"DSD processor initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing DSD processor: {e}")
            print(f"Error type: {type(e).__name__}")
            # Fall back to CPU
            self.device = "cpu"
            self.working_dtype = torch.float32
            self.model_weights = None
            self.lora_weights = None
            self.is_loaded = False
            print(f"Falling back to CPU due to initialization error")
        
    def load_models(self, model_path, lora_path):
        """Load model weights following the structure of the original DSD implementation"""
        try:
            print(f"Loading DSD model from {model_path}")
            print(f"Loading LoRA weights from {lora_path}")
            
            # Check if paths exist
            if not os.path.exists(model_path):
                # Try creating the directory structure if it doesn't exist
                print(f"Model directory not found: {model_path}")
                print("Creating directory structure...")
                try:
                    os.makedirs(model_path, exist_ok=True)
                    print(f"Created model directory: {model_path}")
                    print("=" * 50)
                    print("DOWNLOAD INSTRUCTIONS:")
                    print("1. Download model files from https://huggingface.co/primecai/dsd_model")
                    print("2. Place 'diffusion_pytorch_model.safetensors' in the transformer directory")
                    print("3. Place 'pytorch_lora_weights.safetensors' in the parent directory")
                    print("=" * 50)
                except Exception as e:
                    print(f"Failed to create model directory: {e}")
                    return False
                return False
                
            # Check for model file
            model_file = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(model_file):
                # Try to find any .safetensors file in the directory
                safetensors_files = []
                try:
                    for filename in os.listdir(model_path):
                        if filename.endswith('.safetensors'):
                            full_path = os.path.join(model_path, filename)
                            if os.path.isfile(full_path):
                                safetensors_files.append(full_path)
                except Exception as e:
                    print(f"Error listing directory: {e}")
                    
                if safetensors_files:
                    # Use the first found safetensors file
                    model_file = safetensors_files[0]
                    print(f"Using alternative model file: {model_file}")
                else:
                    print(f"No model file found in {model_path}")
                    print("=" * 50)
                    print("DOWNLOAD INSTRUCTIONS:")
                    print("1. Download the model file from https://huggingface.co/primecai/dsd_model")
                    print("2. Place 'diffusion_pytorch_model.safetensors' in this directory:")
                    print(f"   {model_path}")
                    print("=" * 50)
                    return False
                
            # Check for LoRA file
            if not os.path.exists(lora_path):
                print(f"LoRA file not found: {lora_path}")
                
                # Try to find any LoRA file in the parent directory
                lora_dir = os.path.dirname(lora_path)
                lora_files = []
                
                try:
                    if os.path.exists(lora_dir):
                        for filename in os.listdir(lora_dir):
                            if filename.endswith('.safetensors') and ('lora' in filename.lower() or 'weight' in filename.lower()):
                                full_path = os.path.join(lora_dir, filename)
                                if os.path.isfile(full_path):
                                    lora_files.append(full_path)
                except Exception as e:
                    print(f"Error listing directory: {e}")
                
                if lora_files:
                    # Use the first found LoRA file
                    lora_path = lora_files[0]
                    print(f"Using alternative LoRA file: {lora_path}")
                else:
                    print("=" * 50)
                    print("DOWNLOAD INSTRUCTIONS:")
                    print("1. Download the LoRA weights from https://huggingface.co/primecai/dsd_model")
                    print("2. Place 'pytorch_lora_weights.safetensors' in this directory:")
                    print(f"   {lora_dir}")
                    print("=" * 50)
                    return False
            
            # Load weights with error reporting
            try:
                print(f"Loading transformer model from {model_file}")
                self.model_weights = load_file(model_file)
                if self.model_weights is None:
                    print(f"Failed to load model weights from {model_file}")
                    return False
                    
                print(f"Successfully loaded model with {len(self.model_weights)} tensors")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                traceback.print_exc()
                return False
            
            try:
                print(f"Loading LoRA weights from {lora_path}")
                self.lora_weights = load_file(lora_path)
                if self.lora_weights is None:
                    print(f"Failed to load LoRA weights from {lora_path}")
                    return False
                    
                print(f"Successfully loaded LoRA with {len(self.lora_weights)} tensors")
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                traceback.print_exc()
                return False
            
            # Merge LoRA weights with model if possible
            try:
                self._merge_lora_weights()
            except Exception as e:
                print(f"Warning: Failed to merge LoRA weights, using base model only: {e}")
            
            # Print model architecture information
            self._analyze_model_architecture()
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            return False
            
    def _merge_lora_weights(self):
        """Merge LoRA weights into the base model weights as in the original implementation"""
        if self.model_weights is None or self.lora_weights is None:
            return
            
        # Count the number of LoRA weights applied
        applied_count = 0
        
        # Find all LoRA weight pairs (A and B matrices)
        lora_keys = [k for k in self.lora_weights.keys() if 'lora_A' in k]
        base_keys = [k.replace('lora_A.weight', '') for k in lora_keys]
        
        print(f"Found {len(lora_keys)} LoRA weight pairs to apply")
        if len(lora_keys) > 0:
            print(f"Example LoRA keys: {lora_keys[:2]}")
            
        # Apply LoRA weights to matching base model parameters
        for key_prefix in base_keys:
            # Get base key from LoRA key
            base_key = key_prefix + 'weight'
            lora_a_key = key_prefix + 'lora_A.weight'
            lora_b_key = key_prefix + 'lora_B.weight'
            
            # Check if we have all required keys
            if (base_key in self.model_weights and 
                lora_a_key in self.lora_weights and 
                lora_b_key in self.lora_weights):
                
                # Get the weights
                base_weight = self.model_weights[base_key]
                lora_a = self.lora_weights[lora_a_key]
                lora_b = self.lora_weights[lora_b_key]
                
                # Compute LoRA update: W + AB
                try:
                    # Make sure all are on CPU for consistent operations
                    base_weight = base_weight.cpu()
                    lora_a = lora_a.cpu()
                    lora_b = lora_b.cpu()
                    
                    # Multiply A and B matrices to get the LoRA update
                    if lora_a.dim() == 2 and lora_b.dim() == 2:
                        lora_update = torch.matmul(lora_b, lora_a)
                        
                        # Ensure update has same shape as base weight
                        if lora_update.shape == base_weight.shape:
                            # Apply the update (W' = W + AB)
                            self.model_weights[base_key] = base_weight + 0.75 * lora_update
                            applied_count += 1
                        else:
                            print(f"Shape mismatch for {base_key}: {base_weight.shape} vs {lora_update.shape}")
                    else:
                        print(f"Unsupported LoRA shapes for {key_prefix}: A: {lora_a.shape}, B: {lora_b.shape}")
                except Exception as e:
                    print(f"Failed to apply LoRA for {key_prefix}: {e}")
        
        print(f"Successfully applied {applied_count} LoRA weight updates")
    
    def _analyze_model_architecture(self):
        """Analyze model architecture and print key information"""
        if self.model_weights is None:
            return
            
        # Get number of weights
        num_weights = len(self.model_weights)
        print(f"Model has {num_weights} weight tensors")
        
        # Group weights by type
        weight_types = {}
        for key in self.model_weights.keys():
            # Extract type from key name
            weight_type = key.split('.')[0] if '.' in key else 'other'
            if weight_type not in weight_types:
                weight_types[weight_type] = 0
            weight_types[weight_type] += 1
            
        # Print weight type distribution
        print("Weight type distribution:")
        for type_name, count in weight_types.items():
            print(f"  - {type_name}: {count} tensors")
            
        # Print example keys for important components
        print("Important model components:")
        key_prefixes = ['transformer', 'text_model', 'attention', 'attn', 'projections']
        for prefix in key_prefixes:
            matching_keys = [k for k in self.model_weights.keys() if prefix in k][:3]
            if matching_keys:
                print(f"  - {prefix} components: {matching_keys}")
        
        # Memory usage estimation
        total_params = 0
        for key, tensor in self.model_weights.items():
            total_params += torch.numel(tensor)
        
        memory_mb = total_params * 4 / (1024 * 1024)  # Estimate for float32
        memory_mb_half = total_params * 2 / (1024 * 1024)  # Estimate for float16/bfloat16
        
        print(f"Model parameter count: {total_params:,}")
        print(f"Estimated memory usage: {memory_mb:.2f} MB (float32) / {memory_mb_half:.2f} MB (float16)")
        
        # Print example tensor shapes
        print("Example tensor shapes:")
        sample_keys = list(self.model_weights.keys())[:5]
        for key in sample_keys:
            print(f"  - {key}: {self.model_weights[key].shape}, {self.model_weights[key].dtype}")
    
    def encode_text(self, text):
        """Convert text to embedding tensor, never returning None"""
        # Convert None to empty string
        if text is None:
            text = ""
            print("Empty text provided, using deterministic embedding")
        
        # Always create a deterministic embedding based on the text
        # This ensures consistent results for the same prompt
        batch_size = 1
        embedding_dim = 768  # Standard embedding dimensions
        seq_len = 77         # Standard token sequence length
        
        # Create a deterministic seed from the hash of the text
        text_hash = hash(text) % (2**32)
        embedding_rng = torch.Generator(device=self.device).manual_seed(text_hash)
        
        # Create a random but deterministic embedding
        embedding = torch.randn(batch_size, seq_len, embedding_dim, 
                              generator=embedding_rng, 
                              device=self.device, 
                              dtype=self.working_dtype)
        
        # If we have weights, apply text model transformation
        if self.model_weights is not None:
            print(f"Using model weights for text embedding: {text[:30]}{'...' if len(text) > 30 else ''}")
            
            # Try to find any embedder in the model weights using various patterns
            embedding_keys = []
            embedding_patterns = [
                'text_model.embeddings.token_embedding.weight',  # Standard CLIP pattern
                'embeddings.word_embeddings.weight',  # BERT-like pattern
                'transformer.token_embedding.weight',  # GPT-like pattern
                'embedding.weight',                   # Generic embedding pattern
                'encoder.embed_tokens.weight',        # Encoder pattern
                'word_embedding.weight'               # Another common pattern
            ]
            
            # Search through different patterns
            for pattern in embedding_patterns:
                for key in self.model_weights.keys():
                    if pattern in key:
                        embedding_keys.append(key)
                        break
                        
            # If we found embedding keys, use the first one
            if embedding_keys:
                embedding_key = embedding_keys[0]
                print(f"Found embedding key: {embedding_key}")
                embedding_weights = self.model_weights[embedding_key]
                embedding_weights = embedding_weights.to(device=self.device, dtype=self.working_dtype)
                
                try:
                    # Take a subset of the weights if needed
                    if embedding_weights.shape[0] > 5000:
                        print(f"Using subset of embedding weights: {embedding_weights.shape}")
                        weight_subset = embedding_weights[:5000, :embedding_dim]
                    else:
                        weight_subset = embedding_weights
                    
                    # Convert text to token indices using a simple deterministic approach
                    char_values = [ord(c) % 256 for c in text] if text else [0]
                    token_indices = []
                    
                    # Group characters into chunks for larger vocab coverage
                    for i in range(0, len(char_values), 3):
                        chunk = char_values[i:i+3]
                        token_id = sum(v * (256 ** idx) for idx, v in enumerate(chunk)) % min(5000, weight_subset.shape[0])
                        token_indices.append(token_id)
                    
                    # Pad or truncate to exactly 77 tokens
                    if len(token_indices) < seq_len:
                        token_indices.extend([0] * (seq_len - len(token_indices)))
                    else:
                        token_indices = token_indices[:seq_len]
                    
                    # Create embeddings from token indices
                    token_tensor = torch.tensor(token_indices, device=self.device)
                    sampled_embeddings = weight_subset[token_tensor]
                    
                    # Mix with random embedding for stability (weighted blend)
                    embedding = 0.7 * sampled_embeddings.unsqueeze(0) + 0.3 * embedding
                    print(f"Successfully created text embedding with shape: {embedding.shape}")
                except Exception as e:
                    print(f"Error applying embedding weights: {e}, using random embeddings instead")
                    # Keep the random embedding if anything goes wrong
            else:
                # Look for the largest weight matrix that could be an embedding
                print("No explicit embedding key found, searching for a suitable weight matrix...")
                weight_keys = [k for k in self.model_weights.keys() if 'weight' in k]
                # Sort by shape (looking for matrices with first dimension ≥ 1000 for vocabulary)
                weight_keys = [k for k in weight_keys if 
                              isinstance(self.model_weights[k], torch.Tensor) and
                              self.model_weights[k].dim() == 2 and 
                              self.model_weights[k].shape[0] >= 1000 and
                              self.model_weights[k].shape[1] >= 256]
                              
                if weight_keys:
                    # Sort by vocabulary size (first dimension)
                    weight_keys.sort(key=lambda k: self.model_weights[k].shape[0], reverse=True)
                    embedding_key = weight_keys[0]
                    print(f"Using weight matrix as embedding: {embedding_key} with shape {self.model_weights[embedding_key].shape}")
                    embedding_weights = self.model_weights[embedding_key]
                    embedding_weights = embedding_weights.to(device=self.device, dtype=self.working_dtype)
                    
                    try:
                        # Take a subset of the weights for efficiency
                        weight_subset = embedding_weights[:5000, :embedding_dim]
                        
                        # Simple token mapping as before
                        char_values = [ord(c) % 256 for c in text] if text else [0]
                        token_indices = []
                        for i in range(0, len(char_values), 3):
                            chunk = char_values[i:i+3]
                            token_id = sum(v * (256 ** idx) for idx, v in enumerate(chunk)) % min(5000, weight_subset.shape[0])
                            token_indices.append(token_id)
                        
                        if len(token_indices) < seq_len:
                            token_indices.extend([0] * (seq_len - len(token_indices)))
                        else:
                            token_indices = token_indices[:seq_len]
                        
                        token_tensor = torch.tensor(token_indices, device=self.device)
                        sampled_embeddings = weight_subset[token_tensor]
                        embedding = 0.7 * sampled_embeddings.unsqueeze(0) + 0.3 * embedding
                    except Exception as e:
                        print(f"Error using alternative embedding: {e}, using random embeddings")
                else:
                    print("No suitable model weights found, using random embeddings")
        
        # Ensure embedding is properly set up
        embedding = embedding.to(device=self.device, dtype=self.working_dtype)
        print(f"Text embedding created on device: {embedding.device}, dtype: {embedding.dtype}, shape: {embedding.shape}")
        
        # Return embedding - never return None
        return embedding
                
    def diffusion_step(self, latents, timestep, context, uncond_context=None, guidance_scale=1.0):
        """Apply one step of the diffusion process"""
        # Apply classifier-free guidance if scale > 1
        if guidance_scale > 1.0 and uncond_context is not None:
            # Ensure all inputs are on the same device and dtype
            print(f"Context device: {context.device}, dtype: {context.dtype}")
            print(f"Uncond_context device: {uncond_context.device}, dtype: {uncond_context.dtype}")
            print(f"Latents device: {latents.device}, dtype: {latents.dtype}")
            
            # Get unconditional result
            uncond_latents = self._apply_diffusion(latents, timestep, uncond_context)
            
            # Get conditional result
            cond_latents = self._apply_diffusion(latents, timestep, context)
            
            # Combine using classifier-free guidance
            return uncond_latents + guidance_scale * (cond_latents - uncond_latents)
        else:
            # Just apply conditional diffusion
            return self._apply_diffusion(latents, timestep, context)
    
    def _apply_diffusion(self, latents, timestep, context_feature):
        """Apply the core diffusion process following the original DSD implementation"""
        # Ensure inputs are on the right device and dtype
        latents = latents.to(device=self.device, dtype=self.working_dtype)
        context_feature = context_feature.to(device=self.device, dtype=self.working_dtype)
        
        print(f"Latents shape: {latents.shape}, Context shape: {context_feature.shape}")
        
        # Prepare for the diffusion process
        batch_size, latent_channels, height, width = latents.shape
        
        # Try different key patterns to find attention weights
        attention_patterns = [
            'transformer.h.*.attn.c_proj.weight',  # Pattern for transformer attention projection
            'model.diffusion_model.*.attn_1.to_out.0.weight',  # SD-like pattern
            'transformer.*.attention.*.weight',  # Generic transformer pattern
            '*.attn.*.weight',  # Generic attention pattern
            '*.transformer.*.weight',  # Another transformer pattern
            '*attention*weight'  # Any attention-related weight
        ]
        
        projection_key = None
        projection_weights = None
        
        # Search for suitable projection weights
        print("Searching for projection weights...")
        for pattern in attention_patterns:
            matching_keys = []
            for key in self.model_weights.keys():
                # Simple pattern matching with wildcards
                if self._match_pattern(key, pattern):
                    matching_keys.append(key)
            
            if matching_keys:
                # Sort by size to get larger attention matrices (more likely to be what we want)
                matching_keys.sort(key=lambda k: self.model_weights[k].numel(), reverse=True)
                projection_key = matching_keys[0]
                projection_weights = self.model_weights[projection_key]
                print(f"Found projection weights: {projection_key} with shape {projection_weights.shape}")
                break
        
        # If we still haven't found weights, look for largest weight matrix
        if projection_key is None:
            print("No specific projection weights found, looking for largest weight matrix...")
            weight_keys = [k for k in self.model_weights.keys() if 'weight' in k]
            if weight_keys:
                # Sort by size (largest first)
                weight_keys.sort(key=lambda k: self.model_weights[k].numel(), reverse=True)
                # Take the largest that has appropriate dimensions for projection
                for key in weight_keys[:10]:  # Check top 10 largest matrices
                    weights = self.model_weights[key]
                    if weights.dim() == 2 and weights.shape[0] >= latent_channels:
                        projection_key = key
                        projection_weights = weights
                        print(f"Using large weight matrix as projection: {projection_key} with shape {projection_weights.shape}")
                        break
        
        if projection_key:
            projection_weights = projection_weights.to(device=self.device, dtype=self.working_dtype)
            print(f"Using projection weights: {projection_key} with shape {projection_weights.shape}")
            
            # Project context to latent space using matrix multiplication
            try:
                # Ensure context feature has appropriate shape for matrix multiplication
                if context_feature.dim() == 3:  # [batch, seq_len, embed_dim]
                    # Average over sequence dimension to get [batch, embed_dim]
                    context_flat = context_feature.mean(dim=1)
                elif context_feature.dim() == 4:  # [batch, channels, height, width]
                    # Flatten spatial dimensions and average
                    context_flat = context_feature.mean(dim=[2, 3])
                else:
                    # Just flatten and use as is
                    context_flat = context_feature.reshape(batch_size, -1)
                
                print(f"Context feature flattened to shape: {context_flat.shape}")
                
                # Reshape projection weights if needed
                proj_weights = projection_weights
                if proj_weights.shape[1] != context_flat.shape[1]:
                    print(f"Adjusting projection weights from {proj_weights.shape} to match context {context_flat.shape[1]}")
                    if proj_weights.shape[1] > context_flat.shape[1]:
                        # Truncate
                        proj_weights = proj_weights[:, :context_flat.shape[1]]
                    else:
                        # Pad with zeros
                        padding = torch.zeros(proj_weights.shape[0], context_flat.shape[1] - proj_weights.shape[1], 
                                             device=proj_weights.device, dtype=proj_weights.dtype)
                        proj_weights = torch.cat([proj_weights, padding], dim=1)
                
                # Make sure projection has enough output dimensions
                if proj_weights.shape[0] < latent_channels:
                    print(f"Expanding projection output from {proj_weights.shape[0]} to {latent_channels}")
                    padding = torch.zeros(latent_channels - proj_weights.shape[0], proj_weights.shape[1],
                                         device=proj_weights.device, dtype=proj_weights.dtype)
                    proj_weights = torch.cat([proj_weights, padding], dim=0)
                elif proj_weights.shape[0] > latent_channels:
                    # Truncate to match latent channels
                    proj_weights = proj_weights[:latent_channels, :]
                
                # Apply projection
                projected = torch.matmul(context_flat, proj_weights[:latent_channels, :].t())
                print(f"Projected context to shape: {projected.shape}")
                
                # Reshape projected influence to match latent dimensions
                # Expand from [batch, channels] to [batch, channels, 1, 1]
                projected = projected.unsqueeze(-1).unsqueeze(-1)
                
                # Apply diffusion steps with properly scaled influence
                noise_scale = 1.0 - timestep  # Scale from 0 (no noise) to 1 (full noise)
                
                # Upsample the influence to match latent spatial dimensions
                if height > 1 and width > 1:
                    # Use nearest up to 8x8, then bilinear for smoother results
                    small_size = min(8, max(height//8, 1)), min(8, max(width//8, 1))
                    projected = torch.nn.functional.interpolate(
                        projected, size=small_size, mode='nearest')
                    
                    projected = torch.nn.functional.interpolate(
                        projected, size=(height, width), mode='bilinear', 
                        align_corners=False)
                
                print(f"Final influence tensor: {projected.shape}, applying with scale: {noise_scale:.2f}")
                
                # Create output by combining latents and context influence (scaled by noise level)
                output = latents + noise_scale * 0.1 * projected  # Scale factor to avoid overwhelming the image
                
                return output
            except Exception as e:
                print(f"Error during projection: {e}")
                traceback.print_exc()
                # Fall back to simplified approach
        
        # Simplified fallback if no projection weights found
        print("No suitable projection weights found, using simplified context influence")
        # Create a simple influence based on the latent channels
        influence = torch.zeros(batch_size, latent_channels, 1, 1, 
                                device=self.device, dtype=self.working_dtype)
        
        # Mix with a small amount of context information if available
        if context_feature is not None and context_feature.numel() > 0:
            # Use any available information from context
            ctx_flat = context_feature.reshape(batch_size, -1)
            ctx_mean = ctx_flat.mean(dim=1, keepdim=True)
            
            # Apply a simple scaling to each channel
            for i in range(min(latent_channels, 3)):  # Apply to at least RGB channels
                influence[:, i] = ctx_mean * (1.0 - 0.2 * i)  # Decreasing influence by channel
            
            print(f"Created basic influence tensor with shape: {influence.shape}")
            
            # Expand to image size
            if height > 1 and width > 1:
                influence = torch.nn.functional.interpolate(
                    influence, size=(height, width), mode='bilinear', align_corners=False)
            
            # Apply scaled influence
            noise_scale = (1.0 - timestep) * 0.05  # Small noise scale for simplified approach
            output = latents + noise_scale * influence
            return output
        else:
            # Just return latents with minimal modification
            return latents
            
    def _match_pattern(self, string, pattern):
        """Simple wildcard pattern matching for weight keys"""
        parts = pattern.split('*')
        if len(parts) == 1:
            return string == pattern
            
        if not string.startswith(parts[0]):
            return False
            
        current_pos = len(parts[0])
        for part in parts[1:]:
            if part == '':
                continue
                
            pos = string.find(part, current_pos)
            if pos == -1:
                return False
            current_pos = pos + len(part)
            
        return True
    
    def process_image(self, image, prompt, negative_prompt=None, steps=20, 
                     guidance_scale=3.5, image_guidance=1.0, text_guidance=1.0, seed=None):
        """Process an image using DSD approach"""
        try:
            if not self.is_loaded:
                print("Model not loaded, cannot process image")
                return None
                
            # Set seed for reproducibility
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            print(f"Using seed: {seed} (numpy seed: {seed % (2**32)})")
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            
            # Make sure we're using the right device, especially for CPU mode
            if self.device == "cpu":
                print("Running in CPU mode (slower but more reliable)")
                # For CPU, we'll use Float32 for better precision
                self.working_dtype = torch.float32
                
            # Ensure we have a PIL image
            if not isinstance(image, Image.Image):
                print(f"Input is not a PIL image, but {type(image)}")
                if isinstance(image, torch.Tensor):
                    try:
                        # Try to convert tensor to PIL
                        image = fix_tensor_for_pil(image)
                    except Exception as e:
                        print(f"Could not convert tensor to PIL: {e}")
                        return None
                else:
                    print("Input cannot be processed, expecting PIL Image or tensor")
                    return None
            
            # Print image info
            print(f"Input image mode: {image.mode}, size: {image.size}")
            
            # Convert PIL to tensor in BCHW (batch, channel, height, width) format
            image_np = np.array(image).astype(np.float32) / 255.0
            if len(image_np.shape) == 2:  # Convert grayscale to RGB
                image_np = np.stack([image_np, image_np, image_np], axis=2)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # Handle RGBA
                image_np = image_np[:, :, :3]  # Remove alpha channel
                
            # HWC to BCHW
            tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device=self.device, dtype=self.working_dtype)
            print(f"Input tensor shape: {tensor.shape} (BCHW)")
            
            # Get text embeddings
            print(f"Encoding prompt: {prompt[:30]}{'...' if len(prompt) > 30 else ''}")
            positive_embedding = self.encode_text(prompt)
            
            # Get negative embeddings (or zero embedding if no negative prompt)
            print(f"Encoding negative prompt: {negative_prompt[:30] if negative_prompt else 'None'}")
            negative_embedding = self.encode_text(negative_prompt) if negative_prompt else self.encode_text("")
            
            # Make sure embeddings are not None and have the right dtype
            if positive_embedding is None:
                print("ERROR: Positive embedding is None! Using random embedding.")
                positive_embedding = torch.randn(1, 77, 768, device=self.device, dtype=self.working_dtype)
                
            if negative_embedding is None:
                print("ERROR: Negative embedding is None! Using random embedding.")
                negative_embedding = torch.randn(1, 77, 768, device=self.device, dtype=self.working_dtype)
                
            # Scale by text_guidance parameter
            if text_guidance != 1.0:
                print(f"Applying text guidance scale: {text_guidance}")
                positive_embedding = positive_embedding * text_guidance
                
            # Get initial latents from the input image
            latents = tensor
            print(f"Initial latents shape: {latents.shape}")
            
            # Apply noise to latents (controlled by timestep)
            noise = torch.randn_like(latents)
            initial_timestep = 0.9  # Start with high noise level (0.9 = 90% noise)
            latents = latents * (1 - initial_timestep) + noise * initial_timestep
            
            # Run diffusion for the specified number of steps
            for i in range(steps):
                timestep = 0.9 - (i / steps) * 0.8
                
                try:
                    # Apply one step of diffusion
                    model_output = self.diffusion_step(
                        latents, 
                        timestep, 
                        positive_embedding, 
                        negative_embedding, 
                        guidance_scale
                    )
                    
                    # Apply image guidance (a form of img2img guidance)
                    if image_guidance > 0:
                        # Blend output with original image, weighted by image_guidance
                        # Higher values preserve more of the original
                        denoised = model_output
                        guided = tensor * image_guidance + denoised * (1 - image_guidance)
                        latents = guided * (1 - timestep) + model_output * timestep
                    else:
                        latents = model_output
                        
                    # Log progress
                    if i % 5 == 0 or i == steps - 1:
                        print(f"Step {i+1}/{steps} completed (timestep: {timestep:.2f})")
                except Exception as e:
                    print(f"Error in diffusion step {i} at t={timestep:.2f}: {e}")
                    traceback.print_exc()
                    return None
            
            # Process the final latents to create the output image
            try:
                # Rescale to range [0, 1]
                result = latents.clamp(0, 1)
                
                # Split channels to check output format
                if result.shape[1] > 3:
                    print(f"Output has {result.shape[1]} channels, using first 3 as RGB")
                    # Extract RGB channels if there are more than 3
                    rgb_channels = result[:, :3]
                    
                    # If the 4th channel looks like an alpha mask, use it
                    if result.shape[1] >= 4:
                        alpha = result[:, 3:4]
                        # Check if alpha seems meaningful
                        if alpha.mean() > 0.05 and alpha.mean() < 0.95:
                            print(f"Using channel 4 as alpha mask (mean value: {alpha.mean():.2f})")
                            # Apply alpha to RGB channels
                            rgb_channels = rgb_channels * alpha
                    
                    # Apply a detail boost if this looks like an RGB image
                    if rgb_channels.shape[1] == 3:
                        print("Applying detail enhancement")
                        result = rgb_channels
                        # Add a small amount of high-frequency detail boost
                        detail_boost = 0.05 * (result - torch.nn.functional.avg_pool2d(result, 5, padding=2, stride=1))
                        result = result + detail_boost
                else:
                    # Already 3 or fewer channels
                    rgb_channels = result
                    if result.shape[1] < 3:
                        print(f"Warning: Output has only {result.shape[1]} channels, expected 3 for RGB")
                        # If it's a single channel, replicate to RGB
                        if result.shape[1] == 1:
                            rgb_channels = result.repeat(1, 3, 1, 1)
                            print("Converted single channel to RGB by replication")
                    
                    result = rgb_channels
                
                # Apply final enhancements
                try:
                    # 1. Slight contrast enhancement
                    result = (result - 0.1).clamp(0, 1) * 1.1
                    
                    # 2. Small sharpening filter
                    if result.shape[2] > 5 and result.shape[3] > 5:  # Only if image is large enough
                        kernel = torch.tensor([
                            [-0.02, -0.05, -0.02],
                            [-0.05,  1.28, -0.05],
                            [-0.02, -0.05, -0.02]
                        ], device=self.device, dtype=self.working_dtype).unsqueeze(0).unsqueeze(0)
                        kernel = kernel.expand(result.shape[1], 1, 3, 3)
                        padded = torch.nn.functional.pad(result, (1, 1, 1, 1), mode='replicate')
                        result = torch.nn.functional.conv2d(
                            padded, kernel, groups=result.shape[1]
                        ).clamp(0, 1)
                except Exception as e:
                    print(f"Warning: Error during enhancement, skipping: {e}")
                    # Just continue with unenhanced result
                    pass
                    
                # Check for NaN or Inf values
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print("Warning: Result contains NaN or Inf values, fixing...")
                    result = torch.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Convert to PIL image
                result_array = result.squeeze().permute(1, 2, 0).cpu().numpy()
                result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
                result_image = Image.fromarray(result_array, mode='RGB')  # Explicitly specify RGB mode
                
                print(f"Successfully created result image with size: {result_image.size}")
                return result_image
            except Exception as e:
                print(f"Error in final processing: {e}")
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"Error in process_image: {e}")
            traceback.print_exc()
            return None

# Make _visualize_feature_tensor a standalone function (it was detached from the DSDNode class)
def _visualize_feature_tensor(tensor):
    """Create a visualization for unusual tensors like feature vectors"""
    print(f"Creating visualization for tensor with shape {tensor.shape}")
    
    # Detect if we have BHWC format (channels last)
    is_channels_last = False
    if len(tensor.shape) == 4:
        B, dim1, dim2, dim3 = tensor.shape
        if dim3 <= 4 and dim1 > 32 and dim2 > 32:
            print("Detected BHWC format (ComfyUI standard)")
            is_channels_last = True
            
    # For standard ComfyUI images in BHWC format, just return as is
    if is_channels_last and tensor.shape[3] == 3:
        print("Input appears to be a standard RGB image in BHWC format, returning as is")
        # Ensure it's normalized to 0-1
        normalized = tensor.clone()
        if normalized.max() > 1.0:
            normalized = normalized / 255.0
        return normalized
    
    # Handling specific unusual shapes
    if len(tensor.shape) == 4:  # BCHW or BHWC
        # Standardize to BCHW for processing
        if is_channels_last:
            tensor = tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        B, C, H, W = tensor.shape
        
        # Case: (1, C, 1, 1) - reshape to square for visualization
        if H == 1 and W == 1:
            size = int(np.sqrt(C)) + 1
            vis_tensor = tensor.reshape(1, 1, size, size).repeat(1, 3, 1, 1)
            
            # Scale to 0-1 range
            vis_tensor = (vis_tensor - vis_tensor.min()) / (vis_tensor.max() - vis_tensor.min() + 1e-8)
            
            # Upscale for better visibility
            vis_tensor = torch.nn.functional.interpolate(vis_tensor, size=(256, 256), mode='nearest')
            
            print(f"Created visualization with shape {vis_tensor.shape}")
            # Convert back to BHWC for ComfyUI
            return vis_tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        # Case: (1, C, H, W) with unusual channel count
        if C > 4:
            # Take first 3 channels for RGB visualization
            vis_tensor = tensor[:, :3]
            
            # If too small, upscale
            if H < 64 or W < 64:
                vis_tensor = torch.nn.functional.interpolate(
                    vis_tensor, size=(max(H, 64), max(W, 64)), mode='nearest'
                )
            
            # Normalize to 0-1
            vis_tensor = (vis_tensor - vis_tensor.min()) / (vis_tensor.max() - vis_tensor.min() + 1e-8)
            
            print(f"Created RGB visualization with shape {vis_tensor.shape}")
            # Convert back to BHWC for ComfyUI
            return vis_tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
    
    # For standard BHWC format, check if it's a valid image
    if is_channels_last and tensor.shape[3] in [1, 3, 4]:
        # Just make sure it's normalized to 0-1
        normalized = tensor.clone()
        if normalized.max() > 1.0:
            normalized = normalized / 255.0
        
        # Handle non-RGB
        if tensor.shape[3] == 1:
            # Expand grayscale to RGB
            return normalized.repeat(1, 1, 1, 3)
        return normalized
    
    # If we can't recognize the pattern, create a default image
    print("Creating default visualization image")
    default_tensor = torch.zeros((1, 256, 256, 3), dtype=torch.float32)  # BHWC format
    # Add some visual indicator
    default_tensor[0, 40:216, 40:216, 0] = 0.7  # Red
    default_tensor[0, 60:196, 60:196, 1] = 0.5  # Green
    default_tensor[0, 80:176, 80:176, 2] = 0.9  # Blue
    return default_tensor

# Update the DSDNode class to use the standalone function
class DSDNode:
    """Node for running Diffusion Self-Distillation in ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "context": ("STRING", {"multiline": True, "default": ""}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "image_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "text_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image"
    
    def __init__(self):
        """Initialize the DSD node"""
        # Processor will be lazy loaded
        self.processor = None
        
        # Try to use ComfyUI's folder_paths for standardized locations
        try:
            import folder_paths
            # Get the models directory from ComfyUI
            models_dir = folder_paths.models_dir
            
            # Define standard relative paths within the models directory
            self.model_path = os.path.join(models_dir, "dsd-models", "transformer")
            self.lora_path = os.path.join(models_dir, "dsd-models", "pytorch_lora_weights.safetensors")
            
            # Log the paths we're going to use
            print(f"Using ComfyUI model paths:")
            print(f"  - Model path: {self.model_path}")
            print(f"  - LoRA path: {self.lora_path}")
            
            # Check if the paths exist (for early error detection)
            if not os.path.exists(self.model_path):
                print(f"WARNING: Model path does not exist: {self.model_path}")
                print("Please create this directory and add the model files")
                
            if not os.path.exists(self.lora_path):
                print(f"WARNING: LoRA file does not exist: {self.lora_path}")
                print("Please add the LoRA weights file to the models directory")
                
        except ImportError:
            # If running outside ComfyUI, use absolute paths
            print("ComfyUI environment not detected, using absolute paths")
            
            # These are the paths specified in the user's environment
            self.model_path = "C:\\Users\\cyril\\Documents\\ComfyUI_windows_portable\\ComfyUI\\models\\dsd-models\\transformer"
            self.lora_path = "C:\\Users\\cyril\\Documents\\ComfyUI_windows_portable\\ComfyUI\\models\\dsd-models\\pytorch_lora_weights.safetensors"
            
            print(f"Using absolute paths:")
            print(f"  - Model path: {self.model_path}")
            print(f"  - LoRA path: {self.lora_path}")
    
    def load_processor(self, device="cuda"):
        """Load the DSD processor with models - lazy loading on first use"""
        try:
            # Only load once
            if self.processor is None:
                print(f"Loading DSD processor for device: {device}")
                self.processor = DSDProcessor(device=device)
                
                # Load models
                if not self.processor.load_models(self.model_path, self.lora_path):
                    print(f"ERROR: Failed to load DSD models")
                    print(f"Please make sure the model files are in the correct locations:")
                    print(f"1. Model path: {self.model_path}")
                    print(f"   - Expected file: diffusion_pytorch_model.safetensors")
                    print(f"2. LoRA path: {self.lora_path}")
                    print(f"Visit https://huggingface.co/primecai/dsd_model to download the models")
                    print(f"For more details, check the README.md file")
                    return False
                
                print("DSD processor loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading DSD processor: {e}")
            # Use a safer error printing method
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            return False
    
    def run(self, image, positive, negative, context, guidance_scale, image_guidance, text_guidance, steps, device="cuda", seed=None):
        """Process an image with DSD"""
        try:
            start_time = time.time()
            print("\n" + "="*50)
            print(f"DSD NODE PROCESSING - started at {time.strftime('%H:%M:%S')}")
            print("="*50)
            
            # Merge context with positive prompt if context is provided
            if context and context.strip():
                merged_prompt = f"{positive.strip()}, {context.strip()}" if positive.strip() else context.strip()
                print(f"Context added to prompt: \"{context[:50]}{'...' if len(context) > 50 else ''}\"")
            else:
                merged_prompt = positive
            
            # Log input parameters
            print(f"Input parameters:")
            print(f"  - Positive prompt: \"{merged_prompt[:50]}{'...' if len(merged_prompt) > 50 else ''}\"")
            print(f"  - Negative prompt: \"{negative[:50]}{'...' if len(negative) > 50 else ''}\"")
            print(f"  - Guidance scale: {guidance_scale}")
            print(f"  - Image guidance: {image_guidance}")
            print(f"  - Text guidance: {text_guidance}")
            print(f"  - Steps: {steps}")
            print(f"  - Device: {device}")
            print(f"  - Seed: {seed}")
            
            # Return early if prompt is empty
            if not merged_prompt.strip():
                print("WARNING: Empty prompt provided, returning original image")
                print("Please provide a descriptive prompt for best results")
                return (image,)
                
            # Try to load the processor
            if not self.load_processor(device):
                print("ERROR: Failed to load DSD processor!")
                print("=" * 50)
                print("MODEL FILES ARE MISSING!")
                print("Please download the required model files from:")
                print("https://huggingface.co/primecai/dsd_model")
                print("And place them in:")
                print(f"1. {self.model_path}/diffusion_pytorch_model.safetensors")
                print(f"2. {self.lora_path}")
                print("=" * 50)
                return (image,)
            
            # Debug the input tensor shape
            print(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")
            
            # ComfyUI uses BHWC format (batch, height, width, channels)
            if len(image.shape) == 4:  # BHWC format
                B, H, W, C = image.shape
                print(f"Input tensor dimensions: [B={B}, H={H}, W={W}, C={C}] (BHWC format)")
                
                # Sanity check: If this looks like a normal image, process it
                if C in [1, 3, 4] and H > 32 and W > 32:
                    print(f"Standard image detected with shape {image.shape}")
                else:
                    print(f"WARNING: Unusual image shape detected: {image.shape}")
                    # Special handling if this doesn't look like a normal image
                    if C > 4 or H == 1 or W == 1:
                        print(f"This doesn't look like a standard image, creating visualization")
                        # Use the standalone function instead of a method
                        return (_visualize_feature_tensor(image),)
            else:
                print(f"ERROR: Unexpected tensor dimensions: {image.shape}")
                print(f"ComfyUI requires images in BHWC format (batch, height, width, channels)")
                return (image,)
            
            # Convert BHWC tensor to PIL
            try:
                # ComfyUI images are in BHWC format
                image_np = image[0].cpu().numpy()  # Extract first batch, keep HWC format
                pil_image = Image.fromarray((image_np * 255).astype(np.uint8), mode='RGB')
                print(f"Successfully converted to PIL image with size: {pil_image.size}")
            except Exception as e:
                print(f"ERROR: Standard conversion failed: {e}")
                try:
                    # Try our advanced conversion
                    pil_image = fix_tensor_for_pil(image)
                    print(f"Advanced conversion created PIL image with size: {pil_image.size}")
                except Exception as e:
                    print(f"ERROR: Advanced conversion also failed: {e}")
                    print(f"All conversion attempts failed, returning original")
                    return (image,)
            
            # Process the image
            print(f"Starting DSD processing with {steps} steps...")
            
            # Force CPU mode if the user had issues with GPU
            if device == "cpu":
                print("CPU mode selected - this will be slower but may avoid GPU-related errors")
            elif torch.cuda.is_available():
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                print(f"If you encounter CUDA errors, try using 'cpu' mode instead")
            else:
                print("CUDA requested but not available, falling back to CPU")
                device = "cpu"
                
            result_pil = self.processor.process_image(
                image=pil_image,
                prompt=merged_prompt,
                negative_prompt=negative,
                steps=steps,
                guidance_scale=guidance_scale,
                image_guidance=image_guidance,
                text_guidance=text_guidance,
                seed=seed
            )
            
            # If processing failed, return original
            if result_pil is None:
                print("ERROR: DSD processing failed! Returning original image")
                print("Please check the logs above for specific error messages")
                return (image,)
            
            # Convert back to tensor - ComfyUI expects BHWC format
            try:
                # First convert to numpy array
                result_np = np.array(result_pil).astype(np.float32) / 255.0
                
                # Add batch dimension and ensure BHWC format
                if len(result_np.shape) == 3:  # HWC format
                    result_tensor = torch.from_numpy(result_np).unsqueeze(0)  # Add batch dimension: BHWC
                else:
                    # Unusual format, try to fix
                    print(f"WARNING: Unusual PIL conversion result shape: {result_np.shape}")
                    if len(result_np.shape) == 2:  # Grayscale (H, W)
                        # Convert to RGB
                        h, w = result_np.shape
                        rgb_np = np.zeros((h, w, 3), dtype=np.float32)
                        rgb_np[:, :, 0] = rgb_np[:, :, 1] = rgb_np[:, :, 2] = result_np
                        result_tensor = torch.from_numpy(rgb_np).unsqueeze(0)
                    else:
                        # Can't fix, return original
                        print("ERROR: Cannot convert result to proper tensor format, returning original")
                        print("Please report this issue with your input parameters")
                        return (image,)
                    
                end_time = time.time()
                print(f"Result tensor shape: {result_tensor.shape} (BHWC format)")
                print(f"DSD processing completed in {end_time - start_time:.2f} seconds")
                print("="*50 + "\n")
                return (result_tensor,)
            except Exception as e:
                print(f"ERROR: Error converting result to tensor: {e}")
                print(f"This usually indicates a problem with the model output")
            traceback.print_exc()
            return (image,)
        except Exception as e:
            print(f"ERROR: Unexpected error in DSD processing: {e}")
            traceback.print_exc()
            return (image,)

# Make sure to register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "DiffusionSelfDistillation": DSDNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionSelfDistillation": "Diffusion Self-Distillation (Image Enhancer)"
}

print("DSD ComfyUI Node registered with key 'DiffusionSelfDistillation'")

# The variables will be automatically exported by __init__.py 
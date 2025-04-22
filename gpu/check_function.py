#!/usr/bin/env python3
import os
import sys
import inspect

# Add ComfyUI path
comfyui_path = os.path.abspath("./comfyui/ComfyUI")
sys.path.append(comfyui_path)

try:
    from comfy.sd import load_checkpoint_guess_config
    
    # Print the function signature
    print("Function signature:")
    print(inspect.signature(load_checkpoint_guess_config))
    
    # Print the function docstring
    print("\nFunction documentation:")
    print(load_checkpoint_guess_config.__doc__)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
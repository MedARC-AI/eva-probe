import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np


def dinov3_basic():
    print("=== DINOv3 Basic Example ===")
    
    # model loading
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    print(f"Loading model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {model.config.model_type}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Image size: {model.config.image_size}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    print("Testing inference with dummy image...")
    try:
        dummy_image = Image.new('RGB', (224, 224), color=(128, 64, 192))
        
        # Process the image
        inputs = processor(images=dummy_image, return_tensors="pt")
        print(f"  Input tensor shape: {inputs.pixel_values.shape}")
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check outputs
        print(f"✓ Inference successful")
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            print(f"  Pooler output shape: {outputs.pooler_output.shape}")
            print(f"  Pooler output (CLS token): {outputs.pooler_output.shape}")
        if hasattr(outputs, 'last_hidden_state'):
            print(f"  Last hidden state shape: {outputs.last_hidden_state.shape}")
            print(f"  CLS token shape: {outputs.last_hidden_state[:, 0].shape}")
            print(f"  Patch tokens shape: {outputs.last_hidden_state[:, 1:].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False


def dinov3_variants():
    """different DINOv3 model variants availability."""
    print("=== Testing DINOv3 Model Variants ===")
    
    variants = [
        ("facebook/dinov3-vits16-pretrain-lvd1689m", "ViT-S/16", 384),
        ("facebook/dinov3-vitb16-pretrain-lvd1689m", "ViT-B/16", 768),
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", "ViT-L/16", 1024),
    ]
    
    available_variants = []
    for model_name, display_name, expected_dim in variants:
        try:
            print(f"Testing {display_name} ({model_name})...")
            model = AutoModel.from_pretrained(model_name)
            actual_dim = model.config.hidden_size
            
            print(f"✓ {display_name}")
            print(f"  Expected dim: {expected_dim}, Actual dim: {actual_dim}")
            if actual_dim == expected_dim:
                print(f"  ✓ Dimension matches expected")
            else:
                print(f"  ⚠ Dimension mismatch!")
            
            available_variants.append((model_name, display_name, actual_dim))
            
        except Exception as e:
            print(f"✗ {display_name} failed: {e}")
    
    return available_variants


def eva_compatibility():
    """compatibility with eva-probe patterns."""
    print("=== Testing eva-probe Compatibility ===")
    
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    
    try:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        # batch processing (similar to eva's batch sizes)
        batch_size = 2
        dummy_images = [Image.new('RGB', (224, 224), color=(i*50, 100, 150)) for i in range(batch_size)]
        
        inputs = processor(images=dummy_images, return_tensors="pt")
        print(f"  Batch input shape: {inputs.pixel_values.shape}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # both CLS token (classification) and patch tokens (segmentation) modes
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_features = outputs.pooler_output
            print(f"✓ CLS features (classification): {cls_features.shape}")
        
        if hasattr(outputs, 'last_hidden_state'):
            # CLS token (first token)
            cls_from_hidden = outputs.last_hidden_state[:, 0]
            print(f"✓ CLS from hidden state: {cls_from_hidden.shape}")
            
            # Patch tokens (remaining tokens)
            patch_tokens = outputs.last_hidden_state[:, 1:]
            print(f"✓ Patch tokens (segmentation): {patch_tokens.shape}")
            
            # Calculate expected patch count (14x14 for 224x224 input with patch size 16)
            expected_patches = (224 // 16) ** 2  # 196 patches
            actual_patches = patch_tokens.shape[1]
            print(f"  Expected patches: {expected_patches}, Actual patches: {actual_patches}")
        
        return True
        
    except Exception as e:
        print(f"✗ eva-probe compatibility failed: {e}")
        return False


def main():
    """Run all DINOv3 validation tests."""
    print("DINOv3 Integration Validation for eva-probe")
    print("=" * 50)
    
    # basic functionality
    basic_success = dinov3_basic()
    
    # model variants
    available_variants = dinov3_variants()
    
    # eva compatibility
    eva_success = eva_compatibility()
    
    # Summary
    print("" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Basic functionality: {'✓ PASS' if basic_success else '✗ FAIL'}")
    print(f"Available variants: {len(available_variants)}")
    for model_name, display_name, dim in available_variants:
        print(f"  - {display_name}: {dim} dimensions")
    print(f"eva-probe compatibility: {'✓ PASS' if eva_success else '✗ FAIL'}")
    
    if basic_success and eva_success and len(available_variants) > 0:
        print("🎉 DINOv3 is ready for eva-probe integration!")
    else:
        print("⚠️  Some issues detected. Please review the output above.")


if __name__ == "__main__":
    main()

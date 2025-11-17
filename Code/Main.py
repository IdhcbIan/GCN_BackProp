"""
╭━━━┳╮╭━┳━━━╮╭━━╮╱╱╱╱╱╱╭╮╱╭━━━╮
┃╭━╮┃┃┃╭┫╭━╮┃┃╭╮┃╱╱╱╱╱╱┃┃╱┃╭━╮┃
┃╰━╯┃╰╯╯┃╰━━╮┃╰╯╰┳━━┳━━┫┃╭┫╰━╯┣━┳━━┳━━╮
┃╭╮╭┫╭╮┃╰━━╮┃┃╭━╮┃╭╮┃╭━┫╰╯┫╭━━┫╭┫╭╮┃╭╮┃
┃┃┃╰┫┃┃╰┫╰━╯┃┃╰━╯┃╭╮┃╰━┫╭╮┫┃╱╱┃┃┃╰╯┃╰╯┃
╰╯╰━┻╯╰━┻━━━╯╰━━━┻╯╰┻━━┻╯╰┻╯╱╱╰╯╰━━┫╭━╯
╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱┃┃
╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╰╯
----------------------------------------------------------
// Ian Bezerra - 2025 //
----------------------------------------------------------

END-TO-END TRAINING

Finetunes DINOv2 through differentiable ranking into GCN.
Gradients flow: GCN Loss -> Graph -> Rankings -> Features -> DINOv2
"""

import torch
import config
from Utils import load_classes, load_images, print_experiment_config
from train import run_e2e_experiment


def main():
    """
    Run end-to-end training with DINOv2 finetuning.
    """
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_experiment_config()

    # Load data
    print("Loading data...")
    image_names, labels = load_classes()
    print(f"  Loaded {len(image_names)} images")
    print(f"  Number of classes: {len(set(labels))}")
    print()

    images = load_images(image_names)
    images = images.to(config.DEVICE)
    images.requires_grad_(True)  # Enable gradients on images
    print(f"  Images shape: {images.shape}")
    print(f"  Images require grad: {images.requires_grad}")
    print()

    # Load DINOv2
    print("Loading DINOv2 model...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', config.DINO_MODEL)
    dinov2.to(config.DEVICE)
    print(f"  Model: {config.DINO_MODEL}")
    print(f"  Total parameters: {sum(p.numel() for p in dinov2.parameters()):,}")
    print()

    print("=" * 70)
    print("Starting end-to-end training...")
    print("Gradients will flow: Loss -> GCN -> Rankings -> Features -> DINOv2")
    print("=" * 70)
    print()

    # Run end-to-end experiment
    all_results, overall_accuracy = run_e2e_experiment(dinov2, images, labels)

    print(f"\nFinal E2E accuracy: {overall_accuracy:.4f}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()

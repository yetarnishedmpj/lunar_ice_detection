#!/usr/bin/env python3
"""
Main entry point for the Lunar Ice Detection Pipeline.

Usage:
    python run_pipeline.py --data-dir ./data --output-dir ./output --train

Example:
    # Train the model
    python run_pipeline.py --data-dir ./data --output-dir ./output --train --epochs 50

    # Run inference with pre-trained model
    python run_pipeline.py --data-dir ./data --output-dir ./output --model ./output/model.pth

    # With uncertainty estimation
    python run_pipeline.py --data-dir ./data --output-dir ./output --model ./output/model.pth --uncertainty

    # Use config file
    python run_pipeline.py --config config.yaml
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from pipeline import LunarIceDetectionPipeline, PipelineConfig, create_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Lunar Ice Detection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to YAML configuration file'
    )

    # Data
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data'),
        help='Input data directory containing GeoTIFF files'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./output'),
        help='Output directory for results'
    )

    # Model
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the VAE model'
    )

    parser.add_argument(
        '--model',
        type=Path,
        default=None,
        help='Path to pre-trained model file'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='vae',
        choices=['vae', 'transformer', 'ensemble'],
        help='Model architecture type'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )

    parser.add_argument(
        '--patch-size',
        type=int,
        default=64,
        help='Patch size for VAE input'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--kl-weight',
        type=float,
        default=0.1,
        help='KL divergence weight in VAE loss'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for training (cuda or cpu)'
    )

    # Physics
    parser.add_argument(
        '--no-physics',
        action='store_true',
        help='Disable physics constraints in inference'
    )

    parser.add_argument(
        '--temp-threshold',
        type=float,
        default=110.0,
        help='Temperature threshold for ice stability (K)'
    )

    # Uncertainty
    parser.add_argument(
        '--uncertainty',
        action='store_true',
        help='Enable uncertainty estimation with MC dropout'
    )

    parser.add_argument(
        '--mc-samples',
        type=int,
        default=10,
        help='Number of MC dropout samples for uncertainty'
    )

    # Output
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='lunar_ice',
        help='Prefix for output filenames'
    )

    # Post-processing
    parser.add_argument(
        '--smooth',
        type=float,
        default=1.5,
        help='Gaussian smoothing sigma (pixels). Set to 0 to disable.',
    )
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Disable output smoothing (alias for --smooth 0).',
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip writing the HTML/JSON summary report.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Ice probability threshold used by the summary report.',
    )

    # Advanced
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Use ensemble of models'
    )

    parser.add_argument(
        '--num-models',
        type=int,
        default=5,
        help='Number of models in ensemble'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config_file(config: Dict[str, Any], output_path: Path):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def args_to_config(args) -> PipelineConfig:
    """Convert args to PipelineConfig."""
    sigma = 0.0 if args.no_smooth else float(args.smooth)
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        temp_threshold=args.temp_threshold,
        output_prefix=args.output_prefix,
        smooth_sigma_pixels=sigma,
        generate_report=not args.no_report,
        ice_threshold=args.threshold,
    )

    return config


def main():
    """Main entry point."""
    args = parse_args()

    logger = logging.getLogger(__name__)

    # Load config file if provided
    if args.config:
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)

        logger.info(f"Loading config from {args.config}")
        file_config = load_config_file(args.config)

        # Override with command line args
        for key, value in vars(args).items():
            if value is not None and key not in ['config']:
                file_config[key] = value

        args.data_dir = file_config.get('data_dir', args.data_dir)
        args.output_dir = file_config.get('output_dir', args.output_dir)
        args.epochs = file_config.get('epochs', args.epochs)
        args.batch_size = file_config.get('batch_size', args.batch_size)
        args.patch_size = file_config.get('patch_size', args.patch_size)
        args.learning_rate = file_config.get('learning_rate', args.learning_rate)
        args.kl_weight = file_config.get('kl_weight', args.kl_weight)
        args.temp_threshold = file_config.get('temp_threshold', args.temp_threshold)
        args.output_prefix = file_config.get('output_prefix', args.output_prefix)
        args.uncertainty = file_config.get('uncertainty', args.uncertainty)
        args.mc_samples = file_config.get('mc_samples', args.mc_samples)
        args.no_physics = file_config.get('no_physics', args.no_physics)

    # Verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check data directory exists
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline
    config = args_to_config(args)

    # Create pipeline
    pipeline = LunarIceDetectionPipeline(config, device=args.device)

    # Store extra options
    use_uncertainty = args.uncertainty
    mc_samples = args.mc_samples
    use_physics = not args.no_physics

    try:
        # Run pipeline
        if args.train:
            logger.info("Training mode enabled")

            if args.ensemble:
                logger.info(f"Training ensemble with {args.num_models} models")

            model_path = args.output_dir / 'model.pth'
            ice_prob, indicators, outputs = pipeline.run(
                train=True,
                save_model_path=model_path
            )
        elif args.model and args.model.exists():
            logger.info(f"Loading pre-trained model from {args.model}")

            # Override pipeline's infer method call to add uncertainty
            if args.uncertainty or args.visualize:
                # Load model
                import torch
                from model import create_vae_model
                pipeline.model, _ = create_vae_model(
                    input_channels=7,
                    patch_size=args.patch_size,
                    device=args.device
                )
                pipeline.model.load_state_dict(torch.load(args.model))
                logger.info("Model loaded")

                # Run inference with uncertainty
                feature_tensor = pipeline.load_data()
                ice_prob, indicators = pipeline.infer(
                    feature_tensor,
                    use_physics=use_physics,
                    use_uncertainty=use_uncertainty,
                    mc_samples=mc_samples
                )

                # Generate outputs
                from output import create_output_from_results
                reference_paths = [
                    args.data_dir / fn
                    for fn in config.dataset_files.values()
                ]
                outputs = create_output_from_results(
                    ice_probability=ice_prob,
                    indicators=indicators,
                    reference_paths=reference_paths,
                    output_dir=args.output_dir,
                    prefix=args.output_prefix
                )

                # Apply same smoothing + report logic as the train branch.
                if not args.no_smooth and float(args.smooth) > 0:
                    from postprocessing import smooth_and_summarize
                    ice_prob, indicators, outputs = smooth_and_summarize(
                        ice_probability=ice_prob,
                        indicators=indicators,
                        output_files=outputs,
                        output_dir=args.output_dir,
                        prefix=args.output_prefix,
                        smoothing_sigma=float(args.smooth),
                        threshold=args.threshold,
                        device=args.device,
                    )
                    try:
                        from output import GeoTIFFExporter
                        smoothed_path = (
                            args.output_dir
                            / f"{args.output_prefix}_ice_probability_smoothed.tif"
                        )
                        exporter = GeoTIFFExporter.from_paths(reference_paths)
                        exporter.export_ice_probability(ice_prob, smoothed_path)
                        outputs["ice_probability_smoothed"] = smoothed_path
                    except Exception as exc:
                        logger.warning(f"Could not write smoothed GeoTIFF: {exc}")
            else:
                ice_prob, indicators, outputs = pipeline.run(
                    train=False,
                    save_model_path=args.model
                )
        else:
            logger.error("Please specify --train or --model")
            sys.exit(1)

        # Print output summary
        logger.info("=" * 50)
        logger.info("OUTPUT FILES:")
        for name, path in outputs.items():
            logger.info(f"  {name}: {path}")
        logger.info("=" * 50)

        # Print statistics
        logger.info("=" * 50)
        logger.info("RESULTS SUMMARY:")

        valid_prob = ice_prob[np.isfinite(ice_prob)]
        logger.info(f"  Ice probability - Mean: {valid_prob.mean():.4f}, Max: {valid_prob.max():.4f}")

        if 'uncertainty' in indicators:
            valid_unc = indicators['uncertainty'][np.isfinite(indicators['uncertainty'])]
            logger.info(f"  Uncertainty - Mean: {valid_unc.mean():.4f}")

        high_conf = (valid_prob > 0.5).sum() / len(valid_prob) * 100
        logger.info(f"  High confidence detections: {high_conf:.1f}%")
        logger.info("=" * 50)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()

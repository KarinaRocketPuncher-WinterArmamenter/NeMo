"""
Training script for Conformer-Transformer streaming ASR model.

Example usage:
    python train_conformer_transformer_streaming.py \
        --config-path=../conf/conformer/cache_aware_streaming \
        --config-name=conformer_transformer_bpe_streaming \
        model.train_ds.manifest_filepath=/path/to/train_manifest.json \
        model.validation_ds.manifest_filepath=/path/to/dev_manifest.json \
        model.tokenizer.dir=/path/to/tokenizer_dir \
        trainer.devices=4 \
        trainer.max_epochs=100
"""

import pytorch_lightning as pl
from omegaconf import DictConfig

from nemo.collections.asr.models.conformer_transformer_models import EncDecConformerTransformerModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../conf/conformer/cache_aware_streaming", config_name="conformer_transformer_bpe_streaming")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Initialize trainer
    logging.info("Initializing PyTorch Lightning Trainer")
    trainer = pl.Trainer(**cfg.trainer)
    
    # Setup experiment manager (logging, checkpointing, etc.)
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Initialize model
    logging.info("Initializing Conformer-Transformer model")
    model = EncDecConformerTransformerModel(cfg=cfg.model, trainer=trainer)
    
    # Log model info
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    logging.info(f"Encoder: {model.encoder}")
    logging.info(f"Decoder: {model.decoder}")
    
    # Training
    logging.info("Starting training...")
    trainer.fit(model)
    
    # Testing if test dataset is provided
    if cfg.model.test_ds.manifest_filepath is not None:
        logging.info("Running evaluation on test set...")
        trainer.test(model)
        
    logging.info("Training completed!")


if __name__ == "__main__":
    main()

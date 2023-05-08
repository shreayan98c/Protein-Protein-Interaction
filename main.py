import os
import click
import logging
import transformers
from PPI_Pred.utils import *
from PPI_Pred.dataset import HuRIDataset
from PPI_Pred.model import SimpleLinearModel, SiameseNetwork, SiameseNetworkPretrainer, SiameseNetworkClassification
from PPI_Pred.CL_Attention import CL_AttentionModel, CL_Attention_ConvModel
from PPI_Pred.CrossAttentionModel import *
from PPI_Pred.self_attention import *
from rich.logging import RichHandler
from transformers import EsmTokenizer, EsmModel
from torch.utils.data import DataLoader


@click.group()
def cli():
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
@click.option("--batch-size", default=64)
@click.option("--epochs", default=50)
@click.option("--lr", default=1e-4)
@click.option("--small_subset", default=True)
@click.option("--levels", default=3)
@click.option("--log-interval", default=100, help="Number of batches between logging")
def train(batch_size: int, epochs: int, lr: float, small_subset: bool, levels: int, log_interval: int):
    """
    Train a model.
    :args: batch_size: Batch size
    :args: epochs: Number of epochs
    :args: lr: Learning rate
    :args: small_subset: Use a small subset of the data
    :args: levels: Number of levels in the model
    :args: log_interval: Number of batches between logging
    """
    embed_model_name = "facebook/esm2_t6_8M_UR50D"  # esm2_t12_35M_UR50D, esm2_t6_8M_UR50D
    # esm2_t36_3B_UR50D, esm2_t48_15B_UR50D
    tokenizer = EsmTokenizer.from_pretrained(embed_model_name)
    # esm2_t6_8M_UR50D, esm2_t33_650M_UR50D, esm2_t30_150M_UR50D
    model = EsmModel.from_pretrained(embed_model_name)
    MAX_LEN = 500

    train_dataset = HuRIDataset(tokenizer=tokenizer, model=model, data_split='train', small_subset=small_subset,
                                max_len=MAX_LEN, neg_sample=2)
    test_dataset = HuRIDataset(tokenizer=tokenizer, model=model, data_split='test', small_subset=small_subset,
                               max_len=MAX_LEN, neg_sample=2)
    val_dataset = HuRIDataset(tokenizer=tokenizer, model=model, data_split='valid', small_subset=small_subset,
                              max_len=MAX_LEN, neg_sample=2)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  drop_last=True, shuffle=True)
    validation_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    # Lightning class wraps pytorch model for easier reproducibility
    # simple_cross_attention_model = SelfThenCrossAttentionModel(embed_dim=320, num_heads=5, ff_dim=20, seq_len=MAX_LEN)
    # simple_self_attention_block = SelfAttentionBlock(embed_dim=500, num_heads=5, ff_dim=20)
    # lightning_model_wrapper = LitNonContrastiveClassifier(simple_cross_attention_model, split=True)
    # lightning_model_wrapper = LitNonContrastiveClassifier(simple_cross_attention_block)

    # lightning_model_wrapper = LitNonContrastiveClassifier(SiameseNetwork(d=MAX_LEN), split=True)
    # lightning_model_wrapper = LitContrastivePretrainer(SiameseNetworkPretrainer(d=MAX_LEN))
    lightning_model_wrapper = LitContrastiveClassifier()

    # final model run wrappers
    # lightning_model_wrapper = LitContrastivePretrainer(CL_AttentionModel(embed_dim=320, num_heads=5,
    #                                                                      ff_dim=20, seq_len=MAX_LEN))
    # lightning_model_wrapper = LitContrastivePretrainer(CL_Attention_ConvModel(embed_dim=64, num_heads=8,
    #                                                                           ff_dim=20, seq_len=62, conv_dim=320))

    # Define WandB logger for experiment tracking
    wandb_logger = WandbLogger(project="PPI", name="SiameseClassifier")

    # Define a trainer and fit using it
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/", save_top_k=2, monitor="val_loss")
    trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, callbacks=[checkpoint_callback])

    # if there is a saved checkpoint, place the checkpoint file in the same directory as this file
    # rename the checkpoint file to checkpoint.ckpt so that the trainer can resume from the checkpoint
    if os.path.exists("checkpoint.ckpt"):
        # trainer = pl.Trainer(resume_from_checkpoint="checkpoint.ckpt", max_epochs=epochs, logger=wandb_logger)
        lightning_model_wrapper = lightning_model_wrapper.load_from_checkpoint(
            checkpoint_path="checkpoint.ckpt")
        log.info("Found existing checkpoint.ckpt, loaded model")

    trainer.fit(model=lightning_model_wrapper,
                train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloader)

    # test the model
    trainer.test(model=lightning_model_wrapper, dataloaders=test_dataloader)

    if hasattr(lightning_model_wrapper, 'model') and isinstance(lightning_model_wrapper.model, CL_AttentionModel):
        trainer.save_checkpoint("cl_attention_model.pt", weights_only=True)
        log.info("Model state dict saved for CL Attention model")

    elif hasattr(lightning_model_wrapper, 'model') and isinstance(lightning_model_wrapper.model, CL_Attention_ConvModel):
        trainer.save_checkpoint("siamese_attention_model.pt", weights_only=True)
        log.info("Model state dict saved for Siamese Attention model")

    elif hasattr(lightning_model_wrapper, 'model') and isinstance(lightning_model_wrapper.model, SiameseNetworkPretrainer):
        trainer.save_checkpoint("siamese_pretrained.pt", weights_only=True)
        log.info("Model state dict saved for Siamese model with contrastive loss")

    else:
        trainer.save_checkpoint(
            f"model_weights_{type(lightning_model_wrapper).__name__}.pt", weights_only=True)
        log.info(f"Model state dict saved for {type(lightning_model_wrapper).__name__}")


if __name__ == "__main__":
    cli()

import os
import click
import logging
import transformers
from PPI_Pred.utils import *
from PPI_Pred.dataset import HuRIDataset
from PPI_Pred.model import SimpleLinearModel, SiameseNetwork, SiameseNetworkPretrainer, SiameseNetworkClassification
from PPI_Pred.CL_Attention import CL_AttentionModel, CL_Attention_ConvModel, CL_AttentionModelClassification
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
@click.option("--batch-size", default=4)
@click.option("--epochs", default=50)
@click.option("--lr", default=1e-4)
@click.option("--small_subset", default=True)
@click.option("--levels", default=3)
@click.option("--log-interval", default=100, help="Number of batches between logging")
def test(batch_size: int, epochs: int, lr: float, small_subset: bool, levels: int, log_interval: int):
    """
    Test a model.
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

    test_dataset = HuRIDataset(tokenizer=tokenizer, model=model, data_split='test', small_subset=small_subset,
                               max_len=MAX_LEN, neg_sample=1)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)


    lightning_model_wrapper = LitNonContrastiveClassifier(MultipleSelfThenCrossAttention(embed_dim=320, num_heads=5, ff_dim=20, seq_len=MAX_LEN)) \
        .load_from_checkpoint("./PPI/" + "pd5glx27/checkpoints/epoch=5-step=96.ckpt")

    # Define WandB logger for experiment tracking
    wandb_logger = WandbLogger(project="PPI", name="MultiAttention_On_Test")

    trainer = pl.Trainer(max_epochs=0, logger=wandb_logger)

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

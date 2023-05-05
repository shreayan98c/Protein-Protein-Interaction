import click
import logging
import transformers
from PPI_Pred.utils import *
from PPI_Pred.dataset import HuRIDataset
from PPI_Pred.model import SimpleLinearModel, SiameseNetwork
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
    embed_model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(embed_model_name)  # esm2_t36_3B_UR50D(), esm2_t48_15B_UR50D()
    model = EsmModel.from_pretrained(embed_model_name)  # esm2_t6_8M_UR50D()
    MAX_LEN = 500

    train_dataset = HuRIDataset(tokenizer=tokenizer, model=model, data_split='train', small_subset=small_subset,
                                max_len=MAX_LEN)
    test_dataset = HuRIDataset(tokenizer=tokenizer, model=model, data_split='test', small_subset=small_subset,
                               max_len=MAX_LEN)
    val_dataset = HuRIDataset(tokenizer=tokenizer, model=model, data_split='valid', small_subset=small_subset,
                              max_len=MAX_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    # Lightning class wraps pytorch model for easier reproducibility
    simple_cross_attention_block = CrossAttentionBlock(embed_dim=3000, num_heads=5, ff_dim=20)
    # simple_self_attention_block = SelfAttentionBlock(embed_dim=500, num_heads=5, ff_dim=20)
    lightning_model_wrapper = LitNonContrastiveClassifier(simple_cross_attention_block, split=True)
    # lightning_model_wrapper = LitNonContrastiveClassifier(simple_cross_attention_block)
    # lightning_model_wrapper = LitNonContrastiveClassifier(SiameseNetwork(d=MAX_LEN), split=True)

    # Define WandB logger for experiment tracking
    wandb_logger = WandbLogger(project="PPI", name="self_attention_run")

    # Define a trainer and fit using it
    # trainer = pl.Trainer(max_epochs=1000, logger=wandb_logger)
    # trainer.fit(model=lightning_model_wrapper, train_dataloaders=train_dataloader,
    #             val_dataloaders=validation_dataloader)

    # Define a trainer and fit using it
    trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger)
    trainer.fit(model=lightning_model_wrapper,
                train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloader)

    # test the model
    trainer.test(model=lightning_model_wrapper, dataloaders=test_dataloader)

    # model = SimpleLinearModel(max_len=MAX_LEN, hidden_layers=[50, 25, 3, 1], dropout=0.5)
    # model = SiameseNetwork(d=MAX_LEN)

    # train_simple_linear_model(
    #     model=model,
    #     train_loader=train_dataloader,
    #     test_loader=test_dataloader,
    #     epochs=epochs,
    #     lr=lr,
    #     logging_interval=log_interval,
    # )
    # train_siamese_model(
    #     model=model,
    #     train_loader=train_dataloader,
    #     test_loader=test_dataloader,
    #     epochs=epochs,
    #     lr=lr,
    #     logging_interval=log_interval,
    # )


if __name__ == "__main__":
    cli()

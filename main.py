import click
import logging
import transformers
from PPI_Pred.utils import *
from PPI_Pred.dataset import HuRIDataset
from PPI_Pred.model import SimpleLinearModel
from rich.logging import RichHandler
from transformers import EsmTokenizer
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
@click.option("--batch-size", default=5)
@click.option("--epochs", default=10)
@click.option("--lr", default=1e-3)
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
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")  # esm2_t36_3B_UR50D()

    train_dataset = HuRIDataset(tokenizer=tokenizer, data_split='train', small_subset=small_subset)
    test_dataset = HuRIDataset(tokenizer=tokenizer, data_split='test', small_subset=small_subset)
    val_dataset = HuRIDataset(tokenizer=tokenizer, data_split='valid', small_subset=small_subset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    #Lightning class wraps pytorch model for easier reproducability.: jacky
    lightning_model_wrapper = LitNonContrastiveClassifier(SimpleLinearModel(hidden_layers=[50, 25, 3, 1], dropout=0.3))

    #Define WandB logger for expeperiment tracking
    wandb_logger = WandbLogger(project="PPI",name="Test with metrics")
    
    #Define a trainer and fit using it 
    trainer = pl.Trainer(max_epochs=10,logger = wandb_logger)
    trainer.fit(model=lightning_model_wrapper, train_dataloaders=train_dataloader,val_dataloaders=validation_dataloader)

    #test the model 
    trainer.test(model = lightning_model_wrapper, dataloaders=test_dataloader)


    # train_simple_linear_model(
    #     model=model,
    #     train_loader=train_dataloader,
    #     test_loader=test_dataloader,
    #     epochs=epochs,
    #     lr=lr,
    #     logging_interval=log_interval,
    # )


if __name__ == "__main__":
    cli()

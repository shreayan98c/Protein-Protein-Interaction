import click
import logging
import transformers
from PPI_Pred.model import MyModel
from PPI_Pred.dataset import HuRIDataset
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from transformers import EsmModel, EsmTokenizer


@click.group()
def cli():
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
@click.option("--batch-size", default=32)
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
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")  # esm2_t36_3B_UR50D()

    train_dataset = HuRIDataset(tokenizer=tokenizer, data_split='train', small_subset=small_subset)
    test_dataset = HuRIDataset(tokenizer=tokenizer, data_split='test', small_subset=small_subset)
    val_dataset = HuRIDataset(tokenizer=tokenizer, data_split='valid', small_subset=small_subset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    print(len(train_dataloader))
    print(len(test_dataloader))
    print(len(validation_dataloader))

    # model = MyModel(levels=levels)
    #
    # utils.train(
    #     model=model,
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     lr=lr,
    #     logging_interval=log_interval,
    # )


if __name__ == "__main__":
    cli()

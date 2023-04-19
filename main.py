import click
import logging
import transformers
from PPI_Pred import utils
from PPI_Pred.model import MyModel
from PPI_Pred.dataset import HuRIDataset
from rich.logging import RichHandler
from transformers import EsmModel, EsmConfig, EsmTokenizer


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
@click.option("--levels", default=3)
@click.option("--log-interval", default=100, help="Number of batches between logging")
def train(batch_size: int, epochs: int, lr: float, levels: int, log_interval: int):
    """
    Train a model.
    :args: batch_size: Batch size
    :args: epochs: Number of epochs
    :args: lr: Learning rate
    :args: levels: Number of levels in the model
    :args: log_interval: Number of batches between logging
    """
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")  # esm2_t36_3B_UR50D()

    train_dataset = HuRIDataset(tokenizer=tokenizer, data_split='train')
    test_dataset = HuRIDataset(tokenizer=tokenizer, data_split='test')
    val_dataset = HuRIDataset(tokenizer=tokenizer, data_split='valid')

    print(train_dataset[0])

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

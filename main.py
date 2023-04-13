from PPI_Pred import utils
from PPI_Pred.dataset import MyDataset
from PPI_Pred.model import MyModel
import click
from rich.logging import RichHandler
import logging


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
    train_dataset = MyDataset(train=True)
    test_dataset = MyDataset(train=False)

    model = MyModel(levels=levels)

    utils.train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        logging_interval=log_interval,
    )


if __name__ == "__main__":
    cli()

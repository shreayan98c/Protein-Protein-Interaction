import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from rich.progress import track

log = logging.getLogger(__name__)


def train_simple_linear_model(
        model: nn.Module,
        train_loader,
        test_loader,
        epochs: int,
        lr: float,
        logging_interval: int = 100,
):
    """
    Train a simple linear model.
    :param model: model to train
    :param train_loader: train loader data
    :param test_loader: test loader data
    :param batch_size: batch size
    :param epochs: number of epochs
    :param lr: learning rate
    :param logging_interval: number of batches between logging
    :return:
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    avg_loss = 0.0
    train_acc_store = []
    val_acc_store = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in track(
                enumerate(train_loader), total=len(train_loader), description=f"Train epoch {epoch}"
        ):
            data, target = batch['concatenated_inputs'].float(), batch['label']
            target = target.unsqueeze(1).float()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % logging_interval == 0:
                log.info(f"Epoch {epoch} batch {batch_idx} loss: {avg_loss / logging_interval:.4f}")
                avg_loss = 0.0

        if epoch % 5 == 0:
            torch.save(model, "model.pt")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in track(
                    enumerate(test_loader), total=len(test_loader), description=f"Test epoch {epoch}"
            ):
                data, target = batch['concatenated_inputs'].float(), batch['label']
                target = target.unsqueeze(1).float()
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        log.info(f"Epoch {epoch} accuracy: {correct / total:.4f}")

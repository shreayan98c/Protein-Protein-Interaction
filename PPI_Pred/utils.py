import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from rich.progress import track

log = logging.getLogger(__name__)


def train(
        model: nn.Module,
        train_dataset,
        test_dataset,
        batch_size: int,
        epochs: int,
        lr: float,
        logging_interval: int = 100,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    avg_loss = 0.0

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in track(
                enumerate(train_loader), total=len(train_loader), description=f"Train epoch {epoch}"
        ):
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
            for batch_idx, (data, target) in track(
                    enumerate(test_loader), total=len(test_loader), description=f"Test epoch {epoch}"
            ):
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        log.info(f"Epoch {epoch} accuracy: {correct / total:.4f}")

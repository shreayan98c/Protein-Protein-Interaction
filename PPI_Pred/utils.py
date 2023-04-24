import torch
from torch import nn,optim,utils
from torch.utils.data import DataLoader
import logging
from rich.progress import track
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger


log = logging.getLogger(__name__)


#define a lightnbing Module
class LitNonContrastiveClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.BCELoss()

        #save hyperparams to wandb
        self.save_hyperparameters()
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        data, target = batch['concatenated_inputs'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output = self.model(data)
        loss = self.criterion(output, target)
        # Logging to TensorBoard (if installed) by default
        # log.info("train_loss", loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        
        total = 0
        correct = 0

        with torch.no_grad():
            data, target = batch['concatenated_inputs'].float(), batch['label']
            target = target.unsqueeze(1).float()
            output = self.model(data)
            predicted = torch.round(output.data)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            val_loss = self.criterion(output, target)
            print("hello")
            self.log("val_loss", val_loss)
            self.log("acc",correct/total)
            


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
                predicted = torch.round(output.data)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        log.info(f"Epoch {epoch} accuracy: {correct / total:.4f}")

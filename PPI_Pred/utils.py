import torch
from torch import nn,optim,utils
from torch.utils.data import DataLoader
import logging
from rich.progress import track
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torchmetrics


log = logging.getLogger(__name__)


#define a lightnbing Module to wrap around any non-contrastive classifier we want
#TODO: extend capabilities to contrastive classifiers. 
class LitNonContrastiveClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.BCELoss()

        self.save_hyperparameters()

        #declare metrics to trac
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        data, target = batch['concatenated_inputs'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output = self.model(data)
        loss = self.criterion(output, target)
        predicted = torch.round(output.data)
        self.train_acc(predicted,target)
        # Logging to wandb
        self.log("train_loss", loss, on_step = False, on_epoch= True)
        self.log("train_acc", self.train_acc,on_step=False,on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        
        with torch.no_grad():
            data, target = batch['concatenated_inputs'].float(), batch['label']
            target = target.unsqueeze(1).float()
            output = self.model(data)
            val_loss = self.criterion(output, target)
            predicted = torch.round(output.data)
            self.val_acc(predicted,target)
            self.log("val_loss", val_loss, on_step= False, on_epoch= True)
            self.log("valid_acc",self.val_acc, on_step = False, on_epoch  = True)       

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10 , gamma = 0.9)
        return {"optimizer":optimizer, "lr_scheduler" : {"scheduler" : sch}}
    
    def test_step(self, batch, batch_idx):

        data, target = batch['concatenated_inputs'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output = self.model(data)
        test_loss = self.criterion(output, target)
        predicted = torch.round(output.data)
        self.test_acc(predicted,target)
        self.log("test_loss", test_loss, on_step= False, on_epoch= True)
        self.log("test_acc",self.test_acc, on_step = False, on_epoch  = True)  




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

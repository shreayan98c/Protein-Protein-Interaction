import torch
import torchmetrics
from torch import nn, optim, utils
from torch.nn import BCELoss
from torch.utils.data import DataLoader
import logging
from rich.progress import track
from PPI_Pred.losses import ContrastiveLoss
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from PPI_Pred.model import SimpleLinearModel, SiameseNetwork, SiameseNetworkClassification, SiameseNetworkPretrainer

log = logging.getLogger(__name__)


# define a lightning Module to wrap around any non-contrastive classifier we want
# TODO: extend capabilities to contrastive classifiers.
class LitNonContrastiveClassifier(pl.LightningModule):
    def __init__(self, model, split=True):
        super().__init__()
        self.model = model
        self.criterion = nn.BCELoss()
        self.split = split  # determines if the two sequences should be split on input
        self.save_hyperparameters()

        # declare metrics to track
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        output = None
        if self.split:
            seq1, seq2 = batch['seq1_encoded'].float(), batch['seq2_encoded'].float()
            output = self.model(seq1, seq2)
        else:
            data = batch['concatenated_inputs'].float()
            output = self.model(data)

        target = batch['label']
        target = target.unsqueeze(1).float()
        loss = self.criterion(output, target)
        predicted = torch.round(output.data)
        self.train_acc(predicted, target)
        # Logging to wandb
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    # def validation_step(self,batch,batch_idx): This one has  bugs? - Jacky 

    #     with torch.no_grad():
    #         seq1, seq2, target = batch['seq1_input_ids'].float(), batch['seq1_input_ids'].float(), batch['label']
    #         output = self.model(seq1,seq2)
    #         target = target.unsqueeze(1).float()
    #         val_loss = self.criterion(output, target)
    #         predicted = torch.round(output.data)
    #         self.train_acc(predicted, target)
    #         # Logging to wandb
    #         self.log("train_loss", loss, on_step=False, on_epoch=True)
    #         self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
    #         return loss

    #     # elif isinstance(self.model, SiameseNetwork):
    #     #     seq1, seq2, target = batch['seq1_input_ids'].float(), batch['seq2_input_ids'].float(), batch['label']
    #     #     target = target.unsqueeze(1).float()

    #         # # if using contrastive loss
    #         # output1, output2 = model(seq1, seq2)
    #         # loss = criterion(output1, output2, target, size_average=False)

    #         output = self.model(seq1, seq2)
    #         loss = self.criterion(output, target)
    #         predicted = torch.round(output.data)
    #         self.train_acc(predicted, target)
    #         # Logging to wandb
    #         self.log("train_loss", loss, on_step=False, on_epoch=True)
    #         self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
    #         return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = None
            if self.split:
                seq1, seq2 = batch['seq1_encoded'].float(), batch['seq2_encoded'].float()
                output = self.model(seq1, seq2)
            else:
                data = batch['concatenated_inputs'].float()
                output = self.model(data)

            target = batch['label']
            target = target.unsqueeze(1).float()

            val_loss = self.criterion(output, target)
            predicted = torch.round(output.data)
            self.val_acc(predicted, target)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True)
            self.log("valid_acc", self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sch}}

    def test_step(self, batch, batch_idx):

        output = None
        if self.split:
            seq1, seq2 = batch['seq1_encoded'].float(), batch['seq2_encoded'].float()
            output = self.model(seq1, seq2)
        else:
            data = batch['concatenated_inputs'].float()
            output = self.model(data)

        target = batch['label']
        target = target.unsqueeze(1).float()

        test_loss = self.criterion(output, target)
        predicted = torch.round(output.data)
        self.test_acc(predicted, target)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)


class LitContrastivePretrainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = ContrastiveLoss(margin=1.0)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq2_encoded'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output1, output2 = self.model(seq1, seq2)
        loss = self.criterion(output1, output2, target, size_average=True)
        # Logging to wandb
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq2_encoded'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output1, output2 = self.model(seq1, seq2)
        val_loss = self.criterion(output1, output2, target, size_average=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sch}}

    def test_step(self, batch, batch_idx):
        seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq2_encoded'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output1, output2 = self.model(seq1, seq2)
        test_loss = self.criterion(output1, output2, target, size_average=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)


class LitContrastiveClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.model = model
        # declare metrics to track
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq1_encoded'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output = self.model(seq1, seq2)
        loss = self.criterion(output, target)
        predicted = torch.round(output.data)
        self.train_acc(predicted, target)
        # Logging to wandb
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq1_encoded'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output = self.model(seq1, seq2)
        print(output.shape, target.shape)
        val_loss = self.criterion(output, target)
        predicted = torch.round(output.data)
        self.val_acc(predicted, target)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log("valid_acc", self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sch}}

    def test_step(self, batch, batch_idx):
        seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq2_encoded'].float(), batch['label']
        target = target.unsqueeze(1).float()
        output = self.model(seq1, seq2)
        test_loss = self.criterion(output, target)
        predicted = torch.round(output.data)
        self.test_acc(predicted, target)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)


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


def train_siamese_model(
        model: nn.Module,
        pretrain: bool,
        train_loader,
        test_loader,
        epochs: int,
        lr: float,
        logging_interval: int = 100,
):
    """
    Train a simple linear model.
    :param model: model to train
    :param pretrain: whether to pretrain the model or not
    :param train_loader: train loader data
    :param test_loader: test loader data
    :param epochs: number of epochs
    :param lr: learning rate
    :param logging_interval: number of batches between logging
    :return:
    """
    if pretrain:
        criterion = ContrastiveLoss(margin=1.)
    else:
        criterion = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    avg_loss = 0.0
    train_acc_store = []
    val_acc_store = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in track(
                enumerate(train_loader), total=len(train_loader), description=f"Train epoch {epoch}"
        ):
            seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq2_encoded'].float(), batch['label']
            target = target.unsqueeze(1).float()
            optimizer.zero_grad()

            if pretrain:
                # if using contrastive loss
                output1, output2 = model(seq1, seq2)
                loss = criterion(output1, output2, target, size_average=True)

            else:
                output = model(seq1, seq2)
                loss = criterion(output, target)

            avg_loss += loss.mean().item()
            loss.backward()
            optimizer.step()

            if batch_idx % logging_interval == 0:
                log.info(f"Epoch {epoch} batch {batch_idx} loss: {avg_loss / logging_interval:.4f}")
                avg_loss = 0.0

        if epoch % 5 == 0:
            torch.save(model, "model.pt")

        if not pretrain:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, batch in track(
                        enumerate(test_loader), total=len(test_loader), description=f"Test epoch {epoch}"
                ):
                    seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq2_encoded'].float(), batch['label']
                    target = target.unsqueeze(1).float()

                    # # if using contrastive loss
                    # output1, output2 = model(seq1, seq2)
                    # loss = criterion(output1, output2, target, size_average=False)
                    # predicted = loss > 0.5

                    output = model(seq1, seq2)
                    predicted = torch.round(output.data)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            log.info(f"Epoch {epoch} accuracy: {correct / total:.4f}")

    if pretrain:
        # Saving model state dict and optimizer state dict once training is complete
        torch.save(model.state_dict(), "siamese_pretrained_state_dict.pt")
        torch.save(model, "siamese_pretrained.pt")
        log.info("Model state dict saved for Siamese model with contrastive loss")


def train_siamese_classification_model(
        model: nn.Module,
        train_loader,
        test_loader,
        epochs: int,
        lr: float,
        logging_interval: int = 100,
):
    """
    Train the classification model on top of the pretrained Siamese network.
    :param model: model to train
    :param train_loader: train loader data
    :param test_loader: test loader data
    :param epochs: number of epochs
    :param lr: learning rate
    :param logging_interval: number of batches between logging
    :return: None
    """
    criterion = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    avg_loss = 0.0
    train_acc_store = []
    val_acc_store = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in track(
                enumerate(train_loader), total=len(train_loader), description=f"Train epoch {epoch}"
        ):
            seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq1_encoded'].float(), batch['label']
            target = target.unsqueeze(1).float()
            optimizer.zero_grad()

            # if using contrastive loss
            output = model(seq1, seq2)
            loss = criterion(output, target)

            avg_loss += loss.mean().item()
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
                seq1, seq2, target = batch['seq1_encoded'].float(), batch['seq2_encoded'].float(), batch['label']
                target = target.unsqueeze(1).float()

                output = model(seq1, seq2)
                predicted = torch.round(output.data)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        log.info(f"Epoch {epoch} accuracy: {correct / total:.4f}")


def convert_checkpoint_to_model(ckpt_filepath, save_path):
    """
    Sample usage:

    from PPI_Pred.utils import convert_checkpoint_to_model

    convert_checkpoint_to_model('E:\\BlueJayCodes\\DL\\checkpoints\\SiamesePretrain\\epoch=1-step=2738.ckpt',
                                'E:\\BlueJayCodes\\DL\\checkpoints\\SiamesePretrain\\siamese_pretrained.pt')

    """

    MAX_LEN = 500

    # replace the model wrapper by the type of model you want to load here
    lightning_model_wrapper = LitContrastivePretrainer(SiameseNetworkPretrainer(d=MAX_LEN))
    lightning_model_wrapper = lightning_model_wrapper.load_from_checkpoint(checkpoint_path=ckpt_filepath)
    print('Model checkpoint loaded!')
    torch.save(lightning_model_wrapper.model, save_path)
    print('Model saved!')

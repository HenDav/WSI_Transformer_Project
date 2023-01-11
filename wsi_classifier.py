from argparse import ArgumentParser

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics.functional import auroc

from models.preact_resnet import PreActResNet50


class WsiClassifier(LightningModule):

    # TODO: review hyperparams and add to argparser
    def __init__(self,
                 lr=0.001,
                 num_classes=2,
                 transfer=False,
                 finetune=False,
                 freeze=False,
                 **kwargs):
        super().__init__()

        self.finetune = finetune
        self.save_hyperparameters()

        # TODO: configureable loss function
        self.criterion = nn.CrossEntropyLoss()

        self.model = PreActResNet50(num_classes=num_classes)

        if transfer:  # TODO: transfer / freeze / finetune implementations
            # TODO: loading from different types of checkpoints
            if freeze:
                #  TODO: make configurable or based on specific layer list
                for child in list(self.model.children())[:-1]:
                    for param in child.parameters():
                        param.requires_grad = False
            # self.model.linear = nn.Linear(512 * block.expansion, num_classes)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, preds, scores, y = self.shared_step(batch)

        self.train_acc(preds, y)
        self.log("train/loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("train/acc",
                 self.train_acc,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return {'loss': loss, 'scores': scores.detach(), 'y': y}

    def training_epoch_end(self, outputs):
        self.log('train_auc',
                 auroc(torch.cat([x['scores'] for x in outputs], dim=0),
                       torch.cat([x['y'] for x in outputs])),
                 prog_bar=True,
                 logger=True)

    def validation_step(self, batch, batch_idx):
        loss, preds, scores, y = self.shared_step(batch)

        self.val_acc(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc",
                 self.val_acc,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return {'loss': loss, 'scores': scores, 'y': y}

    def validation_epoch_end(self, outputs):
        self.log('val/auc',
                 auroc(torch.cat([x['scores'] for x in outputs], dim=0),
                       torch.cat([x['y'] for x in outputs])),
                 prog_bar=True,
                 logger=True)

    def test_step(self, batch, batch_idx):
        _, preds, scores, y = self.shared_step(batch)

        self.val_acc(preds, y)
        self.log("test_acc",
                 self.val_acc,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def test_epoch_end(self, outputs):
        # TODO: logging and metrics on validation set for best model
        pass
        # all_slide_scores_max = torch.stack(
        #     [x['slide_score_max'] for x in outputs])
        # all_slide_scores_avg = torch.stack(
        #     [x['slide_score_avg'] for x in outputs])
        # slide_targets = torch.stack([x['y'] for x in outputs]).squeeze()
        # slide_names = [x['slide_name'][0] for x in outputs]
        # slide_scores_orig = [x['slide_score_orig'].item() for x in outputs]
        #
        # # self.log('test_slide_acc_max', torchmetrics.functional.accuracy(all_slide_scores_max, torch.stack([x['y'] for x in outputs]).squeeze()))
        # self.log(
        #     'test_slide_auc_max',
        #     torchmetrics.functional.auroc(all_slide_scores_max, slide_targets))
        #
        # # self.log('test_slide_acc_avg', torchmetrics.functional.accuracy(all_slide_scores_avg, torch.cat([x['y'] for x in outputs]).squeeze()))
        # self.log(
        #     'test_slide_auc_avg',
        #     torchmetrics.functional.auroc(all_slide_scores_avg, slide_targets))
        #
        # # save slide scores
        # df = pd.DataFrame(
        #     data={
        #         'Slide Name': slide_names,
        #         'MilTransformer Score AVG': all_slide_scores_avg.cpu(),
        #         'MilTransformer Score MAX': all_slide_scores_max.cpu(),
        #         'Regular Slide Score': slide_scores_orig,
        #         'Slide Label': slide_targets.cpu()
        #     })
        # if isinstance(self.logger, WandbLogger):
        #     self.logger.log_table(key='slide_scores', dataframe=df)
        # else:
        #     df.to_csv(os.path.join(self.logger.log_dir, 'slide_scores.csv'))
        #

    def shared_step(self, batch):
        # TODO: update based on dataset api
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        # TODO: scores vs preds, also this currently only works for binary classification?
        scores = logits.softmax(1)[:, 1]

        return loss, preds, scores, y

    def configure_optimizers(self):
        # TODO: review optimizer and learning rate scheduler
        if self.finetune:
            pass  # TODO: different learning rates for different layers in finetune, see https://discuss.pytorch.org/t/different-learning-rate-for-a-specific-layer/33670/3
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.lr,
                               betas=(0.9, 0.999),
                               weight_decay=5e-4)
        lr_scheduler = {
            'scheduler':
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(0.3 * self.trainer.max_epochs),
                    int(0.7 * self.trainer.max_epochs)
                ],
                gamma=0.1),
            'interval':
            'epoch'
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr',
                            '--learning_rate',
                            dest='lr',
                            type=float,
                            default=0.001)
        # TODO: transfer / freeze / finetune implemenations and behavior
        parser.add_argument('--transfer', help='', action="store_true")
        parser.add_argument('--finetune', help='', action="store_true")
        parser.add_argument('--freeze', help='', action="store_true")

        return parser

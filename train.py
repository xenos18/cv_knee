from pytorch_lightning import Trainer
from augmentation import image_transformer
from model import MRINet
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


num_epochs = 50
num_classes = 3
learning_rate = 3e-4

if __name__ == '__main__':
    for plane in ['sagittal', 'coronal', 'axial']:
        model = MRINet(num_classes,
                       plane,
                       image_transformer,
                       learning_rate)

        trainer = Trainer(accelerator="gpu",
                          devices=1,
                          max_epochs=num_epochs,
                          default_root_dir="/content/drive/MyDrive/",
                          callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
                          )

        trainer.fit(model)
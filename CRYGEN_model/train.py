import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os, warnings
from . import PARENT_PATH, get_save_path
from .diffusion import CSPDiffusion
from .datamodule import CrystDataModule
from .CRYGEN_configs import get_config
warnings.filterwarnings("ignore")


def get_batch_keys(batch):
    # if torch_geometric<2, batch.keys; else: batch.keys()!
     # don't ask me why, ask the f**king developer of pyg!
    if callable(batch.keys):
        return batch.keys()
    else:
        return batch.keys


def train_loop(model: CSPDiffusion, train_dataloader, val_dataloader, optimizer, scheduler, epochs, early_stopping_patience, device, save_path):
    """Training loop for CSPDiffusion."""
    model.to(device)
    model.train()

    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        early_stopping_flag = False
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()

            # Move batch data to device
            for key in get_batch_keys(batch): 
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            output = model.training_step(batch, batch_idx)

            if output is not None:
                output["loss"].backward()  # Backward pass
                optimizer.step()  # Optimizer step

                train_loss += output["loss"].item()
        train_loss = train_loss / len(train_dataloader)

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                for key in get_batch_keys(batch):
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                output = model.validation_step(batch, batch_idx)
                if output is not None:
                    val_loss += output["val_loss"].item()
        val_loss = val_loss / len(val_dataloader)

        if val_loss < best_loss:
            early_stopping_counter = 0
            best_loss = val_loss
            torch.save(model.state_dict(), save_path + "_best.pt")
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_loss:.4f}, Patience: {early_stopping_counter + 1}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                early_stopping_flag = True
                
        if early_stopping_flag:
            break

        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step(train_loss)


def train(config_name=""):
    os.makedirs(PARENT_PATH / "model_checkpoints", exist_ok=True)

    config = get_config(config_name)
    training_config = config["training_config"]
    datamodule_config = config["datamodule_config"]

    model = CSPDiffusion(
        device=training_config["device"],
        prop_types=datamodule_config["dataset_configs"]["train"].get("prop_types", {"meta_stable": "binary"})
    )

    pretrain_path = training_config.get("pretrain_path", None)
    if pretrain_path is not None:
        model.load_state_dict(
            torch.load(get_save_path(pretrain_path, "model_checkpoints"), weights_only=True),
            strict=False
        )
    save_path = get_save_path(training_config["save_path"], "model_checkpoints")
    data = CrystDataModule(**datamodule_config)
    data.setup(stage="fit")
    train_dataloader, val_dataloaders = \
        data.train_dataloader(), data.val_dataloader() # val and test ARE LIST!!!
    
    print('Data ok!')

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        **training_config["optimizer"] ##lr, beta, etc.
    )

    # Initialize learning rate scheduler
    scheduler = None
    if training_config["use_lr_scheduler"]:
        scheduler = ReduceLROnPlateau(
            optimizer,
            **training_config["lr_scheduler"]
        )

    # Training loop
    train_loop(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloaders[0],
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=training_config["epochs"],
        early_stopping_patience=training_config["early_stopping_patience"],
        device=training_config["device"],
        save_path=save_path
    )
    torch.save(model.state_dict(), save_path + "_last.pt")


if __name__ == "__main__":
    train()

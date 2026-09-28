"""Trainer for sample-based training with distributed support."""
import time

import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

from .dataloader import BranchDataset_Sample, DataLoaderX


class Trainer_Sample:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        batch_size: int = 10,
        device="cpu",
        lr_scheduler=None,
        equilibrium=False,
        grad_clip: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.criterion = criterion
        self.equilibrium = equilibrium
        self.grad_clip = grad_clip
        self.batch_size = batch_size

    def _train_or_eval(self, dataloader, sample_indices, mode="train"):
        assert mode in ["train", "valid", "test"], "Invalid mode."

        is_train = mode == "train"
        is_valid = mode == "valid"
        is_test = mode == "test"

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        sum_squared_error = 0
        sum_squared_true = 0
        print_losses = 0.0

        for data_i in tqdm(dataloader, desc=f"{mode} batches", leave=False):
            data_i = {key: data_i[key].to(self.device, non_blocking=True) for key in data_i.keys()}
            if is_train:
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()

            outputs_y = self.model(data_i, self.device, self.equilibrium)

            if is_train or is_valid:
                loss = self.criterion(outputs_y, data_i["y"])

            # Compute L2 error for all modes
            error = outputs_y - data_i["y"]
            sum_squared_error += (error**2).sum()
            sum_squared_true += (data_i["y"] ** 2).sum()

            if is_train:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if is_train or is_valid:
                print_losses += loss.item()

        if is_train or is_valid:
            avg_print_loss = print_losses / len(dataloader)

        if is_train:
            self.lr_scheduler.step()
            return avg_print_loss, sum_squared_error, sum_squared_true
        elif is_valid:
            return avg_print_loss, sum_squared_error, sum_squared_true
        else:  # test
            return sum_squared_error, sum_squared_true

    @torch.no_grad()
    def valid(self, dataset, sample_indices):
        return self._train_or_eval(dataset, sample_indices, mode="valid")

    @torch.no_grad()
    def test(self, dataset, sample_indices):
        return self._train_or_eval(dataset, sample_indices, mode="test")

    def _train(self, dataset, sample_indices):
        return self._train_or_eval(dataset, sample_indices, mode="train")

    def train(
        self,
        data_train,
        data_valid,
        num_epochs=1000,
        start_epoch=0,
        eval_interval=1,
        save_path=None,
        sample_indices=1000,
        rank=0,
        dyn_checkpoint_prefix="dyn",
        rec_checkpoint_prefix="rec",
        b_checkpoint_prefix="B",
        optimizer_checkpoint_prefix="optimizer",
        save_dyn=True,
        save_rec=True,
        save_B=True,
        save_optimizer=True,
    ):
        train_dataloader = DataLoaderX(
            BranchDataset_Sample(data_train, self.device, sample_indices=sample_indices),
            batch_size=self.batch_size,
            collate_fn=BranchDataset_Sample.collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        valid_dataloader = None
        if data_valid is not None:
            valid_dataloader = DataLoaderX(
                BranchDataset_Sample(data_valid, self.device, sample_indices=sample_indices),
                batch_size=self.batch_size,
                collate_fn=BranchDataset_Sample.collate_fn,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
            )

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.perf_counter()
            loss, train_error, train_sum = self._train(train_dataloader, sample_indices)
            train_error_relative = torch.sqrt(train_error / train_sum).item()

            if valid_dataloader is not None and (epoch + 1) % eval_interval == 0:
                valid_loss, valid_error, valid_sum = self.valid(valid_dataloader, sample_indices)

                loss = torch.tensor([loss], device=self.device)
                valid_loss = torch.tensor([valid_loss], device=self.device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(train_error, op=dist.ReduceOp.AVG)
                    dist.all_reduce(valid_loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(valid_error, op=dist.ReduceOp.AVG)

                train_error_relative = torch.sqrt(train_error / train_sum).item()
                valid_error_relative = torch.sqrt(valid_error / valid_sum).item()

                if rank == 0 and wandb.run is not None:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "train_error": train_error_relative,
                            "valid_loss": valid_loss.item(),
                            "valid_error": valid_error_relative,
                        },
                        step=epoch,
                    )

                if rank == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    epoch_time = time.perf_counter() - epoch_start
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"train_loss={loss.item():.6f} | "
                        f"train_rel={train_error_relative:.6f} | "
                        f"valid_loss={valid_loss.item():.6f} | "
                        f"valid_rel={valid_error_relative:.6f} | "
                        f"lr={lr:.2e} | "
                        f"time={epoch_time:.2f}s"
                    )
            else:
                if rank == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    epoch_time = time.perf_counter() - epoch_start
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"train_loss={loss:.6f} | "
                        f"train_rel={train_error_relative:.6f} | "
                        f"lr={lr:.2e} | "
                        f"time={epoch_time:.2f}s"
                    )

            if (epoch + 1) % 10 == 0 and save_path is not None and rank == 0:
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                if save_dyn:
                    torch.save(model_to_save.dyn.state_dict(), save_path / f"{dyn_checkpoint_prefix}_{epoch}.ckpt")
                if save_rec:
                    torch.save(model_to_save.rec.state_dict(), save_path / f"{rec_checkpoint_prefix}_{epoch}.ckpt")
                if save_B and hasattr(model_to_save, "B"):
                    torch.save(model_to_save.B.state_dict(), save_path / f"{b_checkpoint_prefix}_{epoch}.ckpt")
                if save_optimizer:
                    torch.save(
                        self.optimizer.state_dict(),
                        save_path / f"{optimizer_checkpoint_prefix}_{epoch}.ckpt",
                    )
                print(f"Model saved to {save_path}")

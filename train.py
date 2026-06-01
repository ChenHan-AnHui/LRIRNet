# -*-coding:utf-8-*-
import os
import time
import datetime
import torch
import torch.utils.data as Data
import random
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from train_utils.utils import create_folders, weight_init, save_fixedweight, get_params_groups, time_synchronized
from train_utils.dataset import SirstDataset
from src.LRIRNet import LRIRNet
from train_utils.loss import SoftIoULoss
from train_utils.metric import PD_FA, SigmoidMetric, SamplewiseSigmoidMetric
from thop import profile


class Trainer(object):
    def __init__(self, args, folder, folder_name):
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

        # dataset
        trainset = SirstDataset(args, mode="train")
        valset = SirstDataset(args, mode="val")
        self.train_data_loader = Data.DataLoader(trainset,
                                                 batch_size=args.batch_size,
                                                 num_workers=8,
                                                 shuffle=True,
                                                 pin_memory=True)
        self.val_data_loader = Data.DataLoader(valset,
                                               batch_size=1,  # must be 1
                                               num_workers=8,
                                               pin_memory=True)

        self.model = LRIRNet()
        weight_init(self.model)
        # checkpoint = torch.load("./model_fixed.pth", map_location="cpu")
        # self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        save_fixedweight(self.model, folder)

        # calculate Params Flops
        # flops, params = profile(self.model, inputs=(torch.randn(1, 3, args.base_size, args.base_size, device=self.device), ))
        # print(f"Params: {params / 1e6:.2f}M")
        # print(f"Flops: {flops / 1e9:.2f}G")

        # optimizer
        params_group = get_params_groups(self.model, weight_decay=args.weight_decay)
        self.optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=30)

        # evaluation metrics
        self.PD_FA = PD_FA(1, 10)
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)

        self.best_iou = 0.0
        self.best_nIoU = 0.0

        # SummaryWriter
        self.writer = SummaryWriter(log_dir=folder, flush_secs=5)
        self.writer.add_text(folder_name, "args:%s," % args)

        self.loss, self.lr = 0.0, 0.0

    def training(self, epoch):
        self.model.train()
        losses = []
        tbar = tqdm(self.train_data_loader)
        for image, target in tbar:
            image, target = image.to(self.device), target.to(self.device)
            output = self.model(image)
            loss = SoftIoULoss(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.lr = trainer.optimizer.param_groups[0]["lr"]
            losses.append(loss.item())
            self.loss = np.mean(losses)
            tbar.set_description("Epoch: [%d] lr: %.6f train_loss: %.6f" % (epoch, self.lr, self.loss))

        self.scheduler.step()

        self.writer.add_scalar("train_loss", self.loss, epoch)
        self.writer.add_scalar("lr", self.lr, epoch)

    def validation(self, epoch):
        self.model.eval()
        self.PD_FA.reset()
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        itimes = []
        eval_losses = []
        tbar = tqdm(self.val_data_loader)
        with torch.no_grad():
            for image, target in tbar:
                image, target = image.to(self.device), target.to(self.device)

                t_start = time_synchronized(self.device)
                output = self.model(image)
                t_end = time_synchronized(self.device)
                itimes.append(t_end - t_start)

                loss = SoftIoULoss(output, target)
                eval_losses.append(loss.item())
                self.PD_FA.update(output, target)
                self.iou_metric.update(output, target)
                self.nIoU_metric.update(output, target)
                FA, PD = self.PD_FA.get(len(self.val_data_loader))
                _, IoU = self.iou_metric.get()
                _, nIoU = self.nIoU_metric.get()
                tbar.set_description("Val: val_loss: %.6f itime: %.7f Pd: %.4f Fa: %.4f IoU: %.4f nIoU: %.4f" % (np.mean(eval_losses), np.mean(itimes), PD[0] * 100, FA[0] * 10 ** 6, IoU * 100, nIoU * 100))

        pth_name = "Epoch-%d_Pd-%.4f_Fa-%.4f_IoU-%.4f_nIoU-%.4f.pth" % (epoch, PD[0] * 100, FA[0] * 10 ** 6, IoU * 100, nIoU * 100)
        save_file = self.model.state_dict()

        self.writer.add_scalar("eval_loss", np.mean(eval_losses), epoch)
        self.writer.add_scalar("Pd", PD[0] * 100, epoch)
        self.writer.add_scalar("Fa", FA[0] * 10 ** 6, epoch)
        self.writer.add_scalar("IoU", IoU * 100, epoch)
        self.writer.add_scalar("nIoU", nIoU * 100, epoch)

        results_file = os.path.join(folder, f"results{folder_name}.txt")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、val_loss、lr以及验证集各指标
            write_info = f"[epoch: {epoch}] train_loss: {self.loss:.6f} lr: {self.lr:.6f} val_loss: {np.mean(eval_losses):.6f} itime: {np.mean(itimes):.7f} \n " \
                         f"\t\t\tPD: {PD[0] * 100:.4f} FA: {FA[0] * 10 ** 6:.4f} IoU: {IoU * 100:.4f} nIoU: {nIoU * 100:.4f} \n\n" \

            f.write(write_info)

        # save_best
        if IoU > self.best_iou:
            torch.save(save_file, os.path.join(folder, "best_weights", pth_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(save_file, os.path.join(folder, "best_weights", pth_name))
            self.best_nIoU = nIoU


def parse_args():
    import argparse
    # Setting parameters
    parser = argparse.ArgumentParser(description="Implement of LRIRNet model")

    parser.add_argument("--data-path", type=str, default="./datasets/IRSTD-1k", help="dataset root")
    parser.add_argument("--crop-size", type=int, default=480, help="crop image size")
    parser.add_argument("--base-size", type=int, default=512, help="base image size")

    # Training parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="training device")
    parser.add_argument("--batch-size", type=int, default=12, help="batch_size for training")
    parser.add_argument("--epochs", type=int, default=3000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--wd", "--weight-decay", type=float, default=1e-4,
                        metavar="W", help="weight decay (default: 1e-4)",
                        dest="weight_decay")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # set random seed
    seed = 416
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = parse_args()

    folder, folder_name = create_folders()

    trainer = Trainer(args, folder, folder_name)
    start_time = time.time()

    for epoch in range(args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time:", total_time_str)

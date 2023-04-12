import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as dataloader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import util
import torch.nn as nn
import tqdm
# fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False/
# np.random.seed(SEED)

def valid(model, val_loader, cls_loss_fn, reg_loss_fn, device):
    pass

def train(epochs, model, train_loader, optimizer, cls_loss_fn, reg_loss_fn, device):
    model.train()
    l1_mae_loss = nn.L1Loss()
    for epoch in range(epochs):
        print('*'  * 10 + f' epoch : {epoch} ' + '*'  * 10)
        mask_epoch_loss = 0
        gender_epoch_loss = 0
        age_epoch_loss = 0
        mask_epoch_acc, gender_epoch_acc, age_epoch_mae = 0, 0, 0
        for imgs, (masks, genders, ages) in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            imgs, masks, genders, ages = imgs.to(device), masks.to(device), genders.to(device), ages.float().to(device)
            pred_masks, pred_genders, pred_ages = model(imgs)

            mask_acc = module_metric.accuracy(pred_masks.squeeze(), masks)
            gender_acc = module_metric.accuracy(pred_genders.squeeze(), genders)
            age_mae = l1_mae_loss(pred_ages.squeeze(), ages)

            mask_loss = cls_loss_fn(pred_masks.squeeze(), masks)
            gender_loss = cls_loss_fn(pred_genders.squeeze(), genders)
            age_loss = reg_loss_fn(pred_ages.squeeze(), ages)
            total_loss = mask_loss + gender_loss + age_loss

            mask_epoch_loss += mask_loss.item()
            gender_epoch_loss += gender_loss.item()
            age_epoch_loss += age_loss.item()

            mask_epoch_acc += mask_acc
            gender_epoch_acc += gender_acc
            age_epoch_mae += age_mae.item()            

            total_loss.backward()
            optimizer.step()
        print(f"mask_acc : {mask_epoch_acc / len(train_loader):.4f} | gender_acc : {gender_epoch_acc / len(train_loader):.4f} | age_mae : {age_epoch_mae / len(train_loader):.4f}")                    
        print(f"mask_loss : {mask_epoch_loss / len(train_loader):.4f} | gender_loss : {gender_epoch_loss / len(train_loader):.4f} | age_loss : {age_epoch_loss / len(train_loader):.4f}")

def main(args):
    data_dir = args.data
    epochs = args.epoch
    batch_size = args.batch
    gpus = args.gpus

    # prepare for (multi-device) GPU training
    device, device_ids = util.prepare_device(gpus)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)    

    train_loader = dataloader.MaskDataLoader(data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True)
    model = module_arch.MaskModel()
    model = model.to(device)
    cls_loss_fn = module_loss.cross_entropy_loss
    reg_loss_fn = module_loss.mse_loss
    optimizer = torch.optim.Adam(model.parameters())
    train(epochs, model, train_loader, optimizer, cls_loss_fn, reg_loss_fn, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Level-1 Classification')
    parser.add_argument('-d', '--data', default=None, type=str,
                      help='data path (default: None)')
    parser.add_argument('-e', '--epoch', default=None, type=int,
                      help='number of epochs (default: None)')
    parser.add_argument('-b', '--batch', default=None, type=int,
                        help='number of batch (default: None)')
    parser.add_argument('-g', '--gpus', default=None, type=int,
                      help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    main(args)

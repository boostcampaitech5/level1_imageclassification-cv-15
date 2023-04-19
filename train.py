import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict
from dataset import MaskBaseDataset
from loss import create_criterion
import wandb
from sklearn.metrics import f1_score

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
####

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=EasyDict)

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


# cutmix function
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    if args.wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="level1_be1",
            config=vars(args)
        )
        wandb.run.name = args.name
        wandb.run.save()

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    # num_classes = dataset.num_classes  # 18
    num_classes = 3 * 2 * 3 if not args.multi_label else 3 + 2 + 3

    # -- augmentation
    transform_module = getattr(import_module("augmentation"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset(args.split_stratified)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    lm_criterion = create_criterion('label_smoothing')
    focal_criterion = create_criterion('focal')
    f1_criterion = create_criterion('f1')
    ce_criterion = create_criterion('cross_entropy')


    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4)
    
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=int(len(train_set)/args.batch_size/10),
    #                                             num_training_steps=int(len(train_set) * args.epochs /args.batch_size))

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    

    best_val_acc = 0
    best_val_loss = np.inf
    best_f1_score = 0
    for epoch in range(args.epochs):
        # train loop
        st = time.time()
        model.train()
        loss_value = 0
        matches = 0
        f1_value = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # use mix or not
            if args.cutmix and epoch < args.epochs-10: 
                mix_decision = np.random.rand()
                if mix_decision < args.mix_prob:
                    inputs, mix_labels = cutmix(inputs, labels, 1.0)
                else: 
                  pass
            else: mix_decision = 1

            outs = model(inputs)
            if mix_decision < args.mix_prob:
                loss = criterion(outs, mix_labels[0]) * mix_labels[2] + criterion(outs, mix_labels[1]) * (1 - mix_labels[2])
            else:
                if args.multi_label:
                    (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)
                    mask_labels, gender_labels, age_labels = MaskBaseDataset.decode_multi_class(labels)
                    mask_loss = ce_criterion(mask_outs, mask_labels)
                    gender_loss = ce_criterion(gender_outs, gender_labels)
                    age_loss = f1_criterion(age_outs, age_labels) * 1.5 + lm_criterion(age_outs, age_labels)
                    # mask_loss /= (mask_loss.item() + gender_loss.item() + age_loss.item())
                    # gender_loss /= (mask_loss.item() + gender_loss.item() + age_loss.item())
                    # age_loss /= (mask_loss.item() + gender_loss.item() + age_loss.item())
                    loss = mask_loss + gender_loss + age_loss
                    preds = MaskBaseDataset.encode_multi_class(torch.argmax(mask_outs, -1),
                                                               torch.argmax(gender_outs, -1),
                                                               torch.argmax(age_outs, -1))
    
                else:
                    loss = criterion(outs, labels)
                    preds = torch.argmax(outs, dim=-1)
            # loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            f1_value += f1_score(labels.detach().cpu(), preds.detach().cpu(), average="macro")
            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                train_f1 = f1_value / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 {train_f1:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                f1_value = 0
                if args.multi_label:
                    print(f"mask loss {mask_loss:4.4} || gender loss {gender_loss:4.4} || age loss {age_loss:4.4}")                
                if args.wandb:
                    wandb.log({"Train" : {"acc" : train_acc, "loss" : train_loss, "f1_score" : train_f1}})
                    if args.multi_label:
                        wandb.log({"Train" : {"mask loss" : mask_loss, "gender_loss" : gender_loss, "age_loss" : age_loss}})

        scheduler.step()
        ed = time.time()
        print(f"training time : {(ed - st):.4f}s")

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_mask_loss_items = []
            val_gender_loss_items = []
            val_age_loss_items = []
            val_acc_items = []
            val_f1_scores = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                if args.multi_label:
                    (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)
                    mask_labels, gender_labels, age_labels = MaskBaseDataset.decode_multi_class(labels)
                    mask_loss = ce_criterion(mask_outs, mask_labels).item()
                    gender_loss = ce_criterion(gender_outs, gender_labels).item()
                    age_loss = (f1_criterion(age_outs, age_labels) * 1.5 + lm_criterion(age_outs, age_labels)).item()
                    # mask_loss /= (mask_loss + gender_loss + age_loss)
                    # gender_loss /= (mask_loss + gender_loss + age_loss)
                    # age_loss /= (mask_loss + gender_loss + age_loss)
                    loss_item = mask_loss + gender_loss + age_loss
                    preds = MaskBaseDataset.encode_multi_class(torch.argmax(mask_outs, -1),
                                                               torch.argmax(gender_outs, -1),
                                                               torch.argmax(age_outs, -1))
                else:              
                    loss_item = criterion(outs, labels).item()
                    preds = torch.argmax(outs, dim=-1)
                acc_item = (labels == preds).sum().item()

                f1 = f1_score(labels.detach().cpu(), preds.detach().cpu(), average="macro")
                val_f1_scores.append(f1)
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                if args.multi_label:
                    val_mask_loss_items.append(mask_loss)
                    val_gender_loss_items.append(gender_loss)
                    val_age_loss_items.append(age_loss)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_f1_score = np.sum(val_f1_scores) / len(val_loader)
            if args.multi_label:
                val_mask_loss = np.sum(val_mask_loss_items) / len(val_loader)
                val_gender_loss = np.sum(val_gender_loss_items) / len(val_loader)
                val_age_loss = np.sum(val_age_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            # best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            best_f1_score = max(best_f1_score, val_f1_score)
            if val_loss < best_val_loss:
                print(f"New best model for val loss : {val_loss:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_loss = val_loss
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {val_f1_score:4.2%}|| "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best f1 : {best_f1_score:4.2%}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            if args.multi_label:
                print(f"[Val] mask loss {val_mask_loss:4.4} || gender loss {val_gender_loss:4.4} || age loss {val_age_loss:4.4}")            
            if args.wandb:
                wandb.log({"Valid" : {"acc" : val_acc, "loss" : val_loss, "f1_score" : val_f1_score}, 
                           "valid_examples" : wandb.Image(figure_to_array(figure), caption="valid images")})
                if args.multi_label:
                    wandb.log({"Valid" : {"mask loss" : mask_loss, "gender_loss" : gender_loss, "age_loss" : age_loss}})                
            # scheduler.step(val_loss)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('-c', '--config', default='./config.json', type=str, help='config file path (default: ./config.json)')
    # parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    # parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--valid_batch_size', type=int, default=256, help='input batch size for validing (default: 256)')
    # parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    # parser.add_argument('--criterion', type=str, default='f1', help='criterion type (default: f1)')
    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    # parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    # parser.add_argument('--wandb', type=bool, default=True, help='use wandb')

    # # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    # print(args)
    config = read_json(args.config)
    print(config)

    data_dir = config.data_dir
    model_dir = config.model_dir

    train(data_dir, model_dir, config)

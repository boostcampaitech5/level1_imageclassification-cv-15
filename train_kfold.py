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
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from dataset import TestDataset
import pandas as pd
from util import (seed_everything, increment_path,
                  cutmix, get_lr, grid_image, figure_to_array, read_json, plot_confusion_matrix)

def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader

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

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    lm_criterion = create_criterion('label_smoothing')
    focal_criterion = create_criterion('focal')
    f1_criterion = create_criterion('f1')
    ce_criterion = create_criterion('cross_entropy')

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    skf = StratifiedKFold(n_splits=args.n_splits)
    patience = 10
    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]


    submission = pd.read_csv(os.path.join(args.test_img_root, 'info.csv'))
    image_dir = os.path.join(args.test_img_root, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    test_dataset = TestDataset(image_paths, resize=args.resize)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False
    )    
    oof_pred = None

    # Stratified KFold를 사용해 Train, Valid fold의 Index를 생성합니다.
    # labels 변수에 담긴 클래스를 기준으로 Stratify를 진행합니다. 
    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
        # 생성한 Train, Valid Index를 getDataloader 함수에 전달해 train/valid DataLoader를 생성합니다.
        # 생성한 train, valid DataLoader로 이전과 같이 모델 학습을 진행합니다. 

        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, args.batch_size, multiprocessing.cpu_count() // 2)
        print(f"fold {i} || train images {len(train_loader) * args.batch_size} || valid images {len(val_loader) * args.batch_size}")
        model = model_module(num_classes=num_classes).to(device)
        # model = torch.nn.DataParallel(model)
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        best_val_acc = 0
        best_val_loss = np.inf
        best_val_f1 = 0
        counter = 0
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)    
                        
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
                        f"<Fold : {i}>Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 {train_f1:4.2%} || lr {current_lr}"
                    )

                    loss_value = 0
                    matches = 0
                    f1_value = 0
                    if args.multi_label:
                        print(f"mask loss {mask_loss:4.4} || gender loss {gender_loss:4.4} || age loss {age_loss:4.4}")                
                    if args.wandb:
                        wandb.log({f"fold_{i}" : {"Train" : {"acc" : train_acc, "loss" : train_loss}}})
                        if args.multi_label:
                            wandb.log({f"fold_{i}" : {"Train" : {"mask loss" : mask_loss, "gender_loss" : gender_loss, "age_loss" : age_loss}}})

            scheduler.step()
            ed = time.time()
            print(f"training time : {(ed - st):.4f}s")

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                val_mask_loss_items = []
                val_gender_loss_items = []
                val_age_loss_items = []
                val_f1_scores = []
                figure = None
                val_preds = []
                val_labels = []
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
                    val_labels.append(labels.detach().cpu())
                    val_preds.append(preds.detach().cpu())
                    if args.multi_label:
                        val_mask_loss_items.append(mask_loss)
                        val_gender_loss_items.append(gender_loss)
                        val_age_loss_items.append(age_loss)                
                    # if figure is None:
                    #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    #     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    #     figure = grid_image(
                    #         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    #     )


                cm = confusion_matrix(np.concatenate(val_labels), np.concatenate(val_preds))
                figure_cm = plot_confusion_matrix(cm)
                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_f1_score = np.sum(val_f1_scores) / len(val_loader)
                if args.multi_label:
                    val_mask_loss = np.sum(val_mask_loss_items) / len(val_loader)
                    val_gender_loss = np.sum(val_gender_loss_items) / len(val_loader)
                    val_age_loss = np.sum(val_age_loss_items) / len(val_loader)                
                val_acc = np.sum(val_acc_items) / len(valid_idx)

                # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
                best_val_f1 = max(best_val_f1, val_f1_score)
                best_val_acc = max(best_val_acc, val_acc)                
                if val_loss < best_val_loss:
                    print(f"New best model for val loss : {val_loss:4.4}! saving the model..")
                    torch.save(model.state_dict(), os.path.join(save_dir, f"fold_{i}_best.pth"))
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
                if counter > patience:
                    print("Early Stopping...")
                    break
                print(
                    f"<Fold : {i}>[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.4}, f1: {val_f1_score:4.2%}|| "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.4}, best f1 : {best_val_f1:4.2%}"
                )
                if args.multi_label:
                    print(f"[Val] mask loss {val_mask_loss:4.4} || gender loss {val_gender_loss:4.4} || age loss {val_age_loss:4.4}")                   
                if args.wandb:
                    wandb.log({f"fold_{i}" : {"Valid" : {"acc" : val_acc, "loss" : val_loss, "f1_score" : val_f1_score}, 
                            "valid_confusion_matrix" : wandb.Image(figure_to_array(figure_cm), caption="valid confusion matrix")}})
                    wandb.log({f"fold_{i}" : {"Valid" : {"acc" : val_acc, "loss" : val_loss, "f1_score" : val_f1_score}}})                    
                    if args.multi_label:
                        wandb.log({f"fold_{i}" : {"Valid" : {"mask loss" : mask_loss, "gender_loss" : gender_loss, "age_loss" : age_loss}}})                     
                # scheduler.step(val_loss)
                plt.close()
                print()
        # 각 fold에서 생성된 모델을 사용해 Test 데이터를 예측합니다. 
        all_predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # Test Time Augmentation
                pred = model(images) / 2 # 원본 이미지를 예측하고
                pred += model(torch.flip(images, dims=(-1,))) / 2 # horizontal_flip으로 뒤집어 예측합니다. 
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        # 확률 값으로 앙상블을 진행하기 때문에 'k'개로 나누어줍니다.
        if oof_pred is None:
            oof_pred = fold_pred / args.n_splits
        else:
            oof_pred += fold_pred / args.n_splits

    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label
    
    if args.multi_label:
        (mask_outs, gender_outs, age_outs) = torch.split(torch.from_numpy(oof_pred), [3, 2, 3], dim=1)
        pred = encode_multi_class(torch.argmax(mask_outs, -1), torch.argmax(gender_outs, -1), torch.argmax(age_outs, -1))
    else:
        pred = np.argmax(oof_pred, axis=1)
    submission['ans'] = pred
    # submission['ans'] = np.argmax(oof_pred, axis=1)
    submission.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)

    print('test inference is done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('-c', '--config', default='./config.json', type=str, help='config file path (default: ./config.json)')
    args = parser.parse_args()
    # print(args)
    config = read_json(args.config)
    print(config)

    data_dir = config.data_dir
    model_dir = config.model_dir

    train(data_dir, model_dir, config)

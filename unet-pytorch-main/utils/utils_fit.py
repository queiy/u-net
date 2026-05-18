# import os
# from PIL import Image
# import torch
# from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
#
# from utils.utils import get_lr
# from utils.utils_metrics import f_score
# import numpy as np
# from tqdm import tqdm
#
# def generate_val_predictions(model, dataloader, save_dir, name_list, device='cuda'):
#     os.makedirs(save_dir, exist_ok=True)
#     model.eval()
#     with torch.no_grad():
#         index = 0
#         for images, _ in tqdm(dataloader, desc="Generating validation predictions"):
#             images = images.to(device)
#             outputs = model(images)
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#
#             for i in range(preds.shape[0]):
#                 pred = preds[i].astype(np.uint8)
#                 pred_img = Image.fromarray(pred)
#                 pred_img.save(os.path.join(save_dir, name_list[index] + ".png"))
#                 index += 1
#
# def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
#     total_loss      = 0
#     total_f_score   = 0
#
#     val_loss        = 0
#     val_f_score     = 0
#
#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break
#         imgs, pngs, labels = batch
#         with torch.no_grad():
#             weights = torch.from_numpy(cls_weights)
#             if cuda:
#                 imgs    = imgs.cuda(local_rank)
#                 pngs    = pngs.cuda(local_rank)
#                 labels  = labels.cuda(local_rank)
#                 weights = weights.cuda(local_rank)
#
#         optimizer.zero_grad()
#         if not fp16:
#             #----------------------#
#             #   前向传播
#             #----------------------#
#             outputs = model_train(imgs)
#             #----------------------#
#             #   损失计算
#             #----------------------#
#             if focal_loss:
#                 loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
#             else:
#                 loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
#
#             if dice_loss:
#                 main_dice = Dice_loss(outputs, labels)
#                 loss      = loss + main_dice
#
#             with torch.no_grad():
#                 #-------------------------------#
#                 #   计算f_score
#                 #-------------------------------#
#                 _f_score = f_score(outputs, labels)
#
#             loss.backward()
#             optimizer.step()
#         else:
#             from torch.cuda.amp import autocast
#             with autocast():
#                 #----------------------#
#                 #   前向传播
#                 #----------------------#
#                 outputs = model_train(imgs)
#                 #----------------------#
#                 #   损失计算
#                 #----------------------#
#                 if focal_loss:
#                     loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
#                 else:
#                     loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
#
#                 if dice_loss:
#                     main_dice = Dice_loss(outputs, labels)
#                     loss      = loss + main_dice
#
#                 with torch.no_grad():
#                     #-------------------------------#
#                     #   计算f_score
#                     #-------------------------------#
#                     _f_score = f_score(outputs, labels)
#
#             #----------------------#
#             #   反向传播
#             #----------------------#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#         total_loss      += loss.item()
#         total_f_score   += _f_score.item()
#
#         if local_rank == 0:
#             pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
#                                 'f_score'   : total_f_score / (iteration + 1),
#                                 'lr'        : get_lr(optimizer)})
#             pbar.update(1)
#
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
#
#     model_train.eval()
#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
#         imgs, pngs, labels = batch
#         with torch.no_grad():
#             weights = torch.from_numpy(cls_weights)
#             if cuda:
#                 imgs    = imgs.cuda(local_rank)
#                 pngs    = pngs.cuda(local_rank)
#                 labels  = labels.cuda(local_rank)
#                 weights = weights.cuda(local_rank)
#
#             #----------------------#
#             #   前向传播
#             #----------------------#
#             outputs = model_train(imgs)
#             #----------------------#
#             #   损失计算
#             #----------------------#
#             if focal_loss:
#                 loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
#             else:
#                 loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
#
#             if dice_loss:
#                 main_dice = Dice_loss(outputs, labels)
#                 loss  = loss + main_dice
#             #-------------------------------#
#             #   计算f_score
#             #-------------------------------#
#             _f_score    = f_score(outputs, labels)
#
#             val_loss    += loss.item()
#             val_f_score += _f_score.item()
#
#         if local_rank == 0:
#             pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
#                                 'f_score'   : val_f_score / (iteration + 1),
#                                 'lr'        : get_lr(optimizer)})
#             pbar.update(1)
#
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Validation')
#         loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
#         eval_callback.on_epoch_end(epoch + 1, model_train)
#         print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
#         print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
#
#         #-----------------------------------------------#
#         #   保存权值
#         #-----------------------------------------------#
#         if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
#             torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))
#
#         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
#
#         torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
#
# # def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
# #                   gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes,
# #                   fp16, scaler, save_period, save_dir, local_rank=0):
# #
# #     total_loss = 0
# #     total_f_score = 0
# #     val_loss = 0
# #     val_f_score = 0
# #
# #     if local_rank == 0:
# #         print('Start Train')
# #         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
# #
# #     model_train.train()
# #     for iteration, batch in enumerate(gen):
# #         if iteration >= epoch_step:
# #             break
# #         imgs, pngs, labels = batch
# #         with torch.no_grad():
# #             weights = torch.from_numpy(cls_weights).cuda(local_rank) if cuda else torch.from_numpy(cls_weights)
# #             imgs, pngs, labels = imgs.cuda(local_rank), pngs.cuda(local_rank), labels.cuda(local_rank)
# #
# #         optimizer.zero_grad()
# #         with torch.cuda.amp.autocast(enabled=fp16):
# #             outputs = model_train(imgs)
# #             if focal_loss:
# #                 loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             else:
# #                 loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             if dice_loss:
# #                 main_dice = Dice_loss(outputs, labels)
# #                 loss += main_dice
# #             with torch.no_grad():
# #                 _f_score = f_score(outputs, labels)
# #
# #         if fp16:
# #             scaler.scale(loss).backward()
# #             scaler.step(optimizer)
# #             scaler.update()
# #         else:
# #             loss.backward()
# #             optimizer.step()
# #
# #         total_loss += loss.item()
# #         total_f_score += _f_score.item()
# #         if local_rank == 0:
# #             pbar.set_postfix(**{
# #                 'total_loss': total_loss / (iteration + 1),
# #                 'f_score': total_f_score / (iteration + 1),
# #                 'lr': get_lr(optimizer)
# #             })
# #             pbar.update(1)
# #
# #     if local_rank == 0:
# #         pbar.close()
# #         print('Finish Train\nStart Validation')
# #         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
# #
# #     model_train.eval()
# #     all_preds = []
# #     image_names = []
# #
# #     for iteration, batch in enumerate(gen_val):
# #         if iteration >= epoch_step_val:
# #             break
# #         imgs, pngs, labels = batch
# #         with torch.no_grad():
# #             weights = torch.from_numpy(cls_weights).cuda(local_rank) if cuda else torch.from_numpy(cls_weights)
# #             imgs, pngs, labels = imgs.cuda(local_rank), pngs.cuda(local_rank), labels.cuda(local_rank)
# #             outputs = model_train(imgs)
# #             if focal_loss:
# #                 loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             else:
# #                 loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             if dice_loss:
# #                 loss += Dice_loss(outputs, labels)
# #             _f_score = f_score(outputs, labels)
# #
# #             val_loss += loss.item()
# #             val_f_score += _f_score.item()
# #
# #             # 保存预测图
# #             preds = torch.argmax(outputs, dim=1).cpu().numpy()
# #             all_preds.extend(preds)
# #             batch_filenames = [os.path.splitext(os.path.basename(x))[0] for x in gen_val.dataset.img_paths[iteration * imgs.size(0):(iteration + 1) * imgs.size(0)]]
# #             image_names.extend(batch_filenames)
# #
# #         if local_rank == 0:
# #             pbar.set_postfix(**{
# #                 'val_loss': val_loss / (iteration + 1),
# #                 'f_score': val_f_score / (iteration + 1),
# #                 'lr': get_lr(optimizer)
# #             })
# #             pbar.update(1)
# #
# #     if local_rank == 0:
# #         pbar.close()
# #         print('Finish Validation')
# #         loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
# #         eval_callback.on_epoch_end(epoch + 1, model_train)
# #         print('Epoch:%d/%d\nTotal Loss: %.3f || Val Loss: %.3f' % (
# #             epoch + 1, Epoch, total_loss / epoch_step, val_loss / epoch_step_val))
# #
# #         # 保存模型
# #         if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
# #             torch.save(model.state_dict(), os.path.join(save_dir, f'ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'))
# #         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
# #             print('Save best model to best_epoch_weights.pth')
# #             torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
# #         torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
# #
# #         # ======= 保存预测图像到目录并评估 =======
# #         pred_dir = os.path.join("logs", f"val_pred_epoch_{epoch + 1}")  # ✅ 与 evaluate 保持一致
# #         os.makedirs(pred_dir, exist_ok=True)
# #         print(f"Saving predicted masks to {pred_dir}")
# #         for name, pred in zip(image_names, all_preds):
# #             Image.fromarray(pred.astype(np.uint8)).save(os.path.join(pred_dir, name + ".png"))
# #
# #         # ======= 自动评估 =======
# #         gt_dir = os.path.join('./VOCdevkit/VOC2007/SegmentationClass')
# #         class_names = ['background', 'object']  # 替换为你实际的类别名称
# #         results = evaluate(gt_dir, pred_dir, image_names, num_classes, class_names, save_dir)
# #
# #         print(f"\n[Epoch {epoch + 1}] Evaluation metrics:")
# #         for k, v in results.items():
# #             print(f"  {k}: {v:.4f}")
# #
# #         if loss_history is not None:
# #             for k, v in results.items():
# #                 loss_history.writer.add_scalar(f'val_{k}', v, epoch + 1)
#
#
# from utils.utils_metrics import evaluate
#
#
# # def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
# #                   gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
# #                   save_dir, local_rank=0):
# #     total_loss = 0
# #     total_metrics = {'Precision': 0, 'Recall': 0, 'F1-Score': 0, 'Accuracy': 0, 'IoU': 0}
# #
# #     val_loss = 0
# #     val_metrics = {'Precision': 0, 'Recall': 0, 'F1-Score': 0, 'Accuracy': 0, 'IoU': 0}
# #
# #     if local_rank == 0:
# #         print('Start Train')
# #         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
# #     model_train.train()
# #
# #     for iteration, batch in enumerate(gen):
# #         if iteration >= epoch_step:
# #             break
# #         imgs, pngs, labels = batch
# #         with torch.no_grad():
# #             weights = torch.from_numpy(cls_weights)
# #             if cuda:
# #                 imgs = imgs.cuda(local_rank)
# #                 pngs = pngs.cuda(local_rank)
# #                 labels = labels.cuda(local_rank)
# #                 weights = weights.cuda(local_rank)
# #
# #         optimizer.zero_grad()
# #         if not fp16:
# #             outputs = model_train(imgs)
# #             if focal_loss:
# #                 loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             else:
# #                 loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             if dice_loss:
# #                 main_dice = Dice_loss(outputs, labels)
# #                 loss += main_dice
# #             # metrics = evaluate(outputs, labels)
# #             loss.backward()
# #             optimizer.step()
# #         else:
# #             from torch.cuda.amp import autocast
# #             with autocast():
# #                 outputs = model_train(imgs)
# #                 if focal_loss:
# #                     loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
# #                 else:
# #                     loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
# #                 if dice_loss:
# #                     main_dice = Dice_loss(outputs, labels)
# #                     loss += main_dice
# #                 # metrics = evaluate(outputs, labels)
# #             scaler.scale(loss).backward()
# #             scaler.step(optimizer)
# #             scaler.update()
# #
# #         total_loss += loss.item()
# #         # for key in total_metrics:
# #         #     total_metrics[key] += metrics[key]
# #
# #         if local_rank == 0:
# #             pbar.set_postfix(**{
# #                 'loss': total_loss / (iteration + 1),
# #                 'IoU': total_metrics['IoU'] / (iteration + 1),
# #                 'F1': total_metrics['F1-Score'] / (iteration + 1),
# #                 'Acc': total_metrics['Accuracy'] / (iteration + 1),
# #                 'lr': get_lr(optimizer)
# #             })
# #             pbar.update(1)
# #
# #     if local_rank == 0:
# #         pbar.close()
# #         print('Finish Train')
# #         print('Start Validation')
# #         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
# #
# #     model_train.eval()
# #     for iteration, batch in enumerate(gen_val):
# #         if iteration >= epoch_step_val:
# #             break
# #         imgs, pngs, labels = batch
# #         with torch.no_grad():
# #             weights = torch.from_numpy(cls_weights)
# #             if cuda:
# #                 imgs = imgs.cuda(local_rank)
# #                 pngs = pngs.cuda(local_rank)
# #                 labels = labels.cuda(local_rank)
# #                 weights = weights.cuda(local_rank)
# #
# #             outputs = model_train(imgs)
# #             if focal_loss:
# #                 loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             else:
# #                 loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
# #             if dice_loss:
# #                 main_dice = Dice_loss(outputs, labels)
# #                 loss += main_dice
# #
# #             # metrics = evaluate(outputs, labels)
# #             val_loss += loss.item()
# #             # for key in val_metrics:
# #             #     val_metrics[key] += metrics[key]
# #
# #         if local_rank == 0:
# #             pbar.set_postfix(**{
# #                 'val_loss': val_loss / (iteration + 1),
# #                 'val_IoU': val_metrics['IoU'] / (iteration + 1),
# #                 'val_F1': val_metrics['F1-Score'] / (iteration + 1),
# #                 'val_Acc': val_metrics['Accuracy'] / (iteration + 1),
# #                 'lr': get_lr(optimizer)
# #             })
# #             pbar.update(1)
# #
# #     if local_rank == 0:
# #         pbar.close()
# #         print('Finish Validation')
# #         loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
# #         eval_callback.on_epoch_end(epoch + 1, model_train)
# #
# #         print(f"Epoch {epoch + 1}/{Epoch}")
# #         print(
# #             f"Train - Loss: {total_loss / epoch_step:.4f} | IoU: {total_metrics['IoU'] / epoch_step:.4f} | Acc: {total_metrics['Accuracy'] / epoch_step:.4f} | F1: {total_metrics['F1-Score'] / epoch_step:.4f}")
# #         print(
# #             f"Val   - Loss: {val_loss / epoch_step_val:.4f} | IoU: {val_metrics['IoU'] / epoch_step_val:.4f} | Acc: {val_metrics['Accuracy'] / epoch_step_val:.4f} | F1: {val_metrics['F1-Score'] / epoch_step_val:.4f}")
# #
# #         # 保存模型
# #         if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
# #             torch.save(model.state_dict(), os.path.join(save_dir,
# #                                                         f'ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'))
# #
# #         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
# #             print('Save best model to best_epoch_weights.pth')
# #             torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
# #
# #         torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
#
# def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
#     total_loss      = 0
#     total_f_score   = 0
#
#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break
#         imgs, pngs, labels = batch
#         with torch.no_grad():
#             weights = torch.from_numpy(cls_weights)
#             if cuda:
#                 imgs    = imgs.cuda(local_rank)
#                 pngs    = pngs.cuda(local_rank)
#                 labels  = labels.cuda(local_rank)
#                 weights = weights.cuda(local_rank)
#
#         optimizer.zero_grad()
#         if not fp16:
#
#             outputs = model_train(imgs)
#             if focal_loss:
#                 loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
#             else:
#                 loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
#
#             if dice_loss:
#                 main_dice = Dice_loss(outputs, labels)
#                 loss      = loss + main_dice
#
#             with torch.no_grad():
#                 _f_score = f_score(outputs, labels)
#
#             loss.backward()
#             optimizer.step()
#         else:
#             from torch.cuda.amp import autocast
#             with autocast():
#                 outputs = model_train(imgs)
#
#                 if focal_loss:
#                     loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
#                 else:
#                     loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
#
#                 if dice_loss:
#                     main_dice = Dice_loss(outputs, labels)
#                     loss      = loss + main_dice
#
#                 with torch.no_grad():
#
#                     _f_score = f_score(outputs, labels)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#         total_loss      += loss.item()
#         total_f_score   += _f_score.item()
#
#         if local_rank == 0:
#             pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
#                                 'f_score'   : total_f_score / (iteration + 1),
#                                 'lr'        : get_lr(optimizer)})
#             pbar.update(1)
#
#     if local_rank == 0:
#         pbar.close()
#         loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
#         print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
#         print('Total Loss: %.3f' % (total_loss / epoch_step))
#
#         #-----------------------------------------------#
#         #   保存权值
#         #-----------------------------------------------#
#         if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
#             torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))
#
#         if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
#
#         torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
#
import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():

                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():

                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():

                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(imgs)
                # ----------------------#
                #   损失计算
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
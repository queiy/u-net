# import os
# import matplotlib
# import torch
# import torch.nn.functional as F
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import scipy.signal
# import cv2
# import shutil
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# from .utils import cvtColor, preprocess_input, resize_image
# from .utils_metrics import compute_mIoU
#
# class LossHistory():
#     def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
#         self.log_dir        = log_dir
#         self.val_loss_flag  = val_loss_flag
#
#         self.losses         = []
#         if self.val_loss_flag:
#             self.val_loss   = []
#
#         os.makedirs(self.log_dir)
#         self.writer     = SummaryWriter(self.log_dir)
#         try:
#             dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
#             self.writer.add_graph(model, dummy_input)
#         except:
#             pass
#
#     def append_loss(self, epoch, loss, val_loss = None):
#         if not os.path.exists(self.log_dir):
#             os.makedirs(self.log_dir)
#
#         self.losses.append(loss)
#         if self.val_loss_flag:
#             self.val_loss.append(val_loss)
#
#         with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
#             f.write(str(loss))
#             f.write("\n")
#         if self.val_loss_flag:
#             with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
#                 f.write(str(val_loss))
#                 f.write("\n")
#
#         self.writer.add_scalar('loss', loss, epoch)
#         if self.val_loss_flag:
#             self.writer.add_scalar('val_loss', val_loss, epoch)
#
#         self.loss_plot()
#
#     def loss_plot(self):
#         iters = range(len(self.losses))
#
#         plt.figure()
#         plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
#         if self.val_loss_flag:
#             plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
#
#         try:
#             if len(self.losses) < 25:
#                 num = 5
#             else:
#                 num = 15
#
#             plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
#             if self.val_loss_flag:
#                 plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
#         except:
#             pass
#
#         plt.grid(True)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend(loc="upper right")
#         plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
#         plt.cla()
#         plt.close("all")
# class EvalCallback():
#     def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
#             miou_out_path=".temp_miou_out", eval_flag=True, period=1):
#         super(EvalCallback, self).__init__()
#
#         self.net                = net
#         self.input_shape        = input_shape
#         self.num_classes        = num_classes
#         self.image_ids          = image_ids
#         self.dataset_path       = dataset_path
#         self.log_dir            = log_dir
#         self.cuda               = cuda
#         self.miou_out_path      = miou_out_path
#         self.eval_flag          = eval_flag
#         self.period             = period
#         self.image_ids          = [image_id.split()[0] for image_id in image_ids]
#         self.mious      = [0]
#         self.epoches    = [0]
#         if self.eval_flag:
#             with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
#                 f.write(str(0))
#                 f.write("\n")
#     def get_miou_png(self, image):
#
#         image       = cvtColor(image)
#         orininal_h  = np.array(image).shape[0]
#         orininal_w  = np.array(image).shape[1]
#
#         image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
#
#         image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
#
#         with torch.no_grad():
#             images = torch.from_numpy(image_data)
#             if self.cuda:
#                 images = images.cuda()
#             pr = self.net(images)[0]
#             pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
#             pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
#                     int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
#             pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
#             pr = pr.argmax(axis=-1)
#         image = Image.fromarray(np.uint8(pr))
#         return image
#
#     def on_epoch_end(self, epoch, model_eval):
#         if epoch % self.period == 0 and self.eval_flag:
#             self.net    = model_eval
#             gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
#             pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
#             if not os.path.exists(self.miou_out_path):
#                 os.makedirs(self.miou_out_path)
#             if not os.path.exists(pred_dir):
#                 os.makedirs(pred_dir)
#             print("Get miou.")
#             print(f"[Epoch {epoch}] Start Evaluation.")
#             print(f"Prediction dir: {pred_dir}, GT dir: {gt_dir}, image count: {len(self.image_ids)}")
#             for image_id in tqdm(self.image_ids):
#                 image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
#                 image       = Image.open(image_path)
#                 image       = self.get_miou_png(image)
#                 image.save(os.path.join(pred_dir, image_id + ".png"))
#                 print(f"Saved prediction image: {image_id}.png")
#             print("Calculate miou.")
#             _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
#             print("Per-class IoUs:")
#             for i, iou in enumerate(IoUs):
#                 print(f"  Class {i}: {iou:.4f}")
#             temp_miou = np.nanmean(IoUs) * 100
#             print("Running compute_mIoU...")
#             print(f"IoUs = {IoUs}")
#             print(f"[Epoch {epoch}] Mean IoU: {temp_miou:.2f}%")
#             self.mious.append(temp_miou)
#             self.epoches.append(epoch)
#             with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
#                 f.write(str(temp_miou))
#                 f.write("\n")
#             plt.figure()
#             plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')
#             plt.grid(True)
#             plt.xlabel('Epoch')
#             plt.ylabel('Miou')
#             plt.title('A Miou Curve')
#             plt.legend(loc="upper right")
#             plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
#             plt.cla()
#             plt.close("all")
#             print("Get miou done.")
#             shutil.rmtree(self.miou_out_path)
import os
import matplotlib
import torch
import torch.nn.functional as F
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []

        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")
from utils.utils_metrics import compute_mIoU, per_Accuracy, show_results,f_score_numpy,per_class_Accuracy
import pandas as pd

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1, name_classes=None):
        super(EvalCallback, self).__init__()

        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        self.name_classes       = name_classes if name_classes else [str(i) for i in range(num_classes)]

        self.mious              = [0]
        self.epoches            = [0]

        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            print("Get miou.")
            print(f"[Epoch {epoch}] Start Evaluation.")
            print(f"Prediction dir: {pred_dir}, GT dir: {gt_dir}, image count: {len(self.image_ids)}")

            for image_id in tqdm(self.image_ids):
                image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
                image = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                print(f"Saved prediction image: {image_id}.png")

            print("Calculate miou.")
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, self.name_classes)
            acc = per_Accuracy(hist)

            # 读取标签和预测用于 f_score
            preds_all = []
            labels_all = []
            for image_id in self.image_ids:
                pred_path = os.path.join(pred_dir, image_id + ".png")
                label_path = os.path.join(gt_dir, image_id + ".png")
                # pred = np.array(Image.open(pred_path), dtype=np.uint8)
                # label_img = Image.open(label_path)
                pred = np.array(Image.open(pred_path).convert("L"), dtype=np.uint8)
                # label_img = Image.open(label_path).convert("L")
                # label_img = np.array(label_img, dtype=np.uint8)
                #
                # # 自动调整标签大小与预测一致
                # if label_img.size != pred.shape[::-1]:
                #     print(f"[Warning] Resizing label {image_id} from {label_img.size} to {pred.shape[::-1]}")
                #     label_img = label_img.resize(pred.shape[::-1], Image.NEAREST)
                #
                # label = np.array(label_img, dtype=np.uint8)
                label_img = Image.open(label_path).convert("L")

                if label_img.size != pred.shape[::-1]:
                    print(f"[Warning] Resizing label {image_id} from {label_img.size} to {pred.shape[::-1]}")
                    label_img = label_img.resize(pred.shape[::-1], Image.NEAREST)

                label = np.array(label_img, dtype=np.uint8)
                # 检查并跳过尺寸不一致的（极端情况）
                if pred.shape != label.shape or pred.ndim != label.ndim:
                    print(f"[SKIPPED] {image_id} - shape mismatch:")
                    print(f"    pred.shape: {pred.shape}, dtype: {pred.dtype}")
                    print(f"    label.shape: {label.shape}, dtype: {label.dtype}")
                    continue

                preds_all.append(pred)
                labels_all.append(label)
            min_shape = preds_all[0].shape
            consistent_preds = []
            consistent_labels = []

            for i, (p, l) in enumerate(zip(preds_all, labels_all)):
                if p.shape != min_shape or l.shape != min_shape:
                    print(f"[SKIPPED-BEFORE-STACK] Sample {i} has inconsistent shape: pred {p.shape}, label {l.shape}")
                    continue
                consistent_preds.append(p)
                consistent_labels.append(l)

            try:
                preds_all = np.stack(preds_all, axis=0)
                labels_all = np.stack(labels_all, axis=0)
                fscore, class_fscore = f_score_numpy(labels_all, preds_all, return_per_class=True)
            except Exception as e:
                print(f"[Error] Stacking failed: {e}")
                fscore = 0.0
                class_fscore = {i: 0.0 for i in range(self.num_classes)}

            per_class_acc = per_class_Accuracy(hist)

            print(f"[Epoch {epoch}] Pixel Accuracy: {acc * 100:.2f}%")
            print(f"[Epoch {epoch}] f_score: {fscore:.4f}")

            temp_miou = np.nanmean(IoUs) * 100
            print(f"[Epoch {epoch}] Mean IoU: {temp_miou:.2f}%")

            self.mious.append(temp_miou)
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")

            # 保存曲线
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='train miou')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            # 可视化保存图
            show_results(self.log_dir, hist, IoUs, PA_Recall, Precision, self.name_classes, tick_font_size=12)

            # 保存到 Excel
            rows = []
            for i in range(self.num_classes):
                f1_score = class_fscore.get(i, 0.0)
                if f1_score is None:
                    f1_score = 0.0
                    print(f"[Warning] Class {i} not found in predictions or labels. Setting F1 to 0.")
                print(f"[DEBUG] Raw F1 for class {i}: {class_fscore.get(i, 0)}")
                print(f"===> {i}: IoU-{IoUs[i] * 100:.2f}; "
                      f"Recall-{PA_Recall[i] * 100:.2f}; "
                      f"Precision-{Precision[i] * 100:.2f}; "
                      f"PixelAcc-{per_class_acc[i] * 100:.2f}; "
                      f"F1-{f1_score * 100:.2f}")
                rows.append({
                    "Epoch": epoch,
                    "Class": self.name_classes[i],
                    "IoU": IoUs[i],
                    "Recall": PA_Recall[i],
                    "Precision": Precision[i],
                    "Pixel Accuracy": per_class_acc[i],
                    "F-Score": f1_score
                })
            rows.append({
                "Epoch": epoch,
                "Class": "Overall",
                "IoU": np.nanmean(IoUs),
                "Recall": np.nanmean(PA_Recall),
                "Precision": np.nanmean(Precision),
                "Pixel Accuracy": acc,
                "F-Score": fscore
            })
            excel_path = os.path.join(self.log_dir, "metrics_summary.xlsx")
            df = pd.DataFrame(rows)
            if os.path.exists(excel_path):
                df_old = pd.read_excel(excel_path)
                df = pd.concat([df_old, df], ignore_index=True)
            df.to_excel(excel_path, index=False)
            print(f"[Epoch {epoch}] Saved metrics to {excel_path}")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
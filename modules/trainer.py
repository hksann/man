import logging
import os
from abc import abstractmethod
import time
import torch
import random
import numpy as np
from numpy import inf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .base_cmn import BaseCMN
from torch.cuda.amp import autocast, GradScaler
from googletrans import Translator
from .loss import compute_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 设置参与训练的GPU
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        
        # 创建 AMP GradScaler 对象
        self.scaler = GradScaler()
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
                
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There's no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            msg = "Warning: The number of GPU's configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
            print(msg)
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

class Trainer(BaseTrainer):
    
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader, lr_ve, lr_ed, step_size, gamma):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.train_loss_history = []
        self.val_metrics_history = {}
        self.test_metrics_history = {}
        
        self.lr_ve = lr_ve  # 设置 lr_ve 属性
        self.lr_ed = lr_ed  # 设置 lr_ed 属性
    
    def translate_to_chinese(self, text_to_translate):
        # 创建一个Translator对象
        translator = Translator()
        # 执行文本翻译，从英文到中文
        translated_text = translator.translate(text_to_translate, src='en', dest='zh-cn').text
        return translated_text
    
    def _train_epoch(self, epoch):
        print(logging.getLogger().getEffectiveLevel())
        train_loss = 0
        self.model.train()

        train_start_time = time.time()

        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)

            with autocast():
                output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)

            train_loss += loss.item()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if batch_idx % 50 == 0:
                loss_message = '[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'.format(epoch, self.epochs, batch_idx, len(self.train_dataloader), train_loss / (batch_idx + 1))
                print(loss_message)

        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print(f"[{epoch}/{self.epochs}] End of epoch average training loss: {log['train_loss']:.5f}. Epoch Time: {train_time:.2f} seconds")

        # 验证部分
        val_start_time = time.time()
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

                # 计算损失
                loss = self.criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
                val_loss += loss.item()

                # 在第一个批次中随机选择两张图片并打印信息
                if batch_idx == 0:
                    random_indices = np.random.choice(len(images_id), 2, replace=False)  # 随机选择两个索引
                    for idx in random_indices:
                        image_name = images_id[idx]  # 图片id
                        translated_inference_text = self.translate_to_chinese(reports[idx])  # 翻译推理文本
                        translated_ground_truth_text = self.translate_to_chinese(ground_truths[idx])  # 翻译地面真实文本

                        # 修改后的验证集打印代码
                        print(f"Validation Set - Image Name: {image_name}")
                        print(f"\033[31mValidation Set - Inference Text: {reports[idx]} (Translated: {translated_inference_text})\033[0m")  # 红色
                        print(f"Validation Set - Ground Truth Text: {ground_truths[idx]} (Translated: {translated_ground_truth_text})")

            val_loss_avg = val_loss / len(self.val_dataloader)
            log['val_loss'] = val_loss_avg

            # 使用 ReduceLROnPlateau 更新学习率
            self.lr_scheduler.step(val_loss_avg)
            
            # 获取更新后的学习率并以科学记数法打印
            current_lr_ve = self.optimizer.param_groups[0]['lr']
            current_lr_ed = self.optimizer.param_groups[1]['lr']
            print(f"Updated Learning Rate for Visual Extractor (VE): {current_lr_ve:.2e}")
            print(f"Updated Learning Rate for Encoder-Decoder (ED): {current_lr_ed:.2e}")


            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        
        # 验证结束后获取时间戳
        val_end_time = time.time()
        # 计算验证过程花费的时间
        val_time = val_end_time - val_start_time
        
        test_start_time = time.time()  # 记录测试开始时间
        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

                if batch_idx == 0:
                    random_indices = np.random.choice(len(images_id), 2, replace=False)
                    for idx in random_indices:
                        image_name = images_id[idx]
                        translated_inference_text = self.translate_to_chinese(reports[idx])
                        translated_ground_truth_text = self.translate_to_chinese(ground_truths[idx])

                        # 测试集打印代码
                        print(f"Test Set - Image Name: {image_name}")
                        print(f"\033[31mTest Set - Inference Text: {reports[idx]} (Translated: {translated_inference_text})\033[0m")  # 红色
                        print(f"Test Set - Ground Truth Text: {ground_truths[idx]} (Translated: {translated_ground_truth_text})")

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)}, {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        test_end_time = time.time()  # 记录测试结束时间
        test_time = test_end_time - test_start_time  # 计算测试时间

        # 每五个 epoch 绘制和保存指标图表
        if epoch % 1 == 0:
            title = f"Epoch: {epoch}, LR VE: {self.lr_ve}, LR ED: {self.lr_ed}"
            self._plot_metrics(epoch, title)

        # 打印训练、验证和测试时间
        print(f"[{epoch}/{self.epochs}] Train Time: {train_time:.2f} seconds")
        print(f"[{epoch}/{self.epochs}] Validation Time: {val_time:.2f} seconds")
        print(f"[{epoch}/{self.epochs}] Test Time: {test_time:.2f} seconds")

        return log
    
    def generate_image_path(self, epoch, prefix=''):
        """
        生成图表的保存路径。
        :param epoch: 当前的训练周期。
        :param prefix: 文件名前缀。
        :return: 图表的文件路径。
        """
        image_name = f'{prefix}_epoch_{epoch}.png'
        return os.path.join('/kaggle/working/', image_name)
    
    def _plot_metrics(self, epoch, title=''):
        """
        绘制并保存指标图表。
        :param epoch: 当前的训练周期。
        :param title: 图表的标题，用于区分不同类型的图表。
        """
        try:
            # 删除上一个图片文件（如果存在）
            previous_image_path = self.generate_image_path(epoch - 1, title)
            if os.path.exists(previous_image_path):
                os.remove(previous_image_path)

            # 创建图表
            plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(2, 2)

            # 绘制训练损失
            plt.subplot(gs[0, :])
            plt.plot(self.train_loss_history, label='Training Loss', color='blue')
            plt.title(f'Training Loss (Epoch {epoch})', fontsize=20)
            plt.xlabel('Batch', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.legend(fontsize=12)
            plt.grid(True)

            # 绘制验证和测试指标
            self.plot_individual_metrics(self.val_metrics_history, 'Validation', gs[1, 0], epoch)
            self.plot_individual_metrics(self.test_metrics_history, 'Test', gs[1, 1], epoch)

            plt.suptitle(title, fontsize=20)  # 使用提供的标题
            plt.tight_layout()

            # 保存图表为图片文件
            image_path = self.generate_image_path(epoch, title)
            plt.savefig(image_path, dpi=300)
            plt.close()  # 关闭图形以节约内存
        except Exception as e:
            print(f"Error occurred during plotting: {e}")
        finally:
            plt.close()

    def plot_individual_metrics(self, metrics_history, title_prefix, subplot_index, epoch):
        """
        绘制单个指标图表。
        :param metrics_history: 指标历史数据。
        :param title_prefix: 图表标题前缀。
        :param subplot_index: 子图索引。
        :param epoch: 当前的训练周期。
        """
        if metrics_history is not None:
            plt.subplot(subplot_index)
            colors = sns.color_palette('husl', n_colors=len(metrics_history))
            linestyles = ['-', '--', '-.', ':']
            markers = ['o', 's', 'D', '^', 'v', '<', '>']
            for i, (metric_name, metric_values) in enumerate(metrics_history.items()):
                plt.plot(metric_values, label=f'{title_prefix} {metric_name}', color=colors[i], linestyle=linestyles[i % len(linestyles)], marker=markers[i % len(markers)])
            plt.title(f'{title_prefix} Metrics (Epoch {epoch})', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Metric Value', fontsize=15)
            plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True)

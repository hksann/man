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
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR
from .optimizers import build_lr_scheduler, GradualWarmupScheduler

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
        # 初始化学习率调度器
        # 根据参数选择适当的学习率调度器
        if args.lr_scheduler == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=args.reduce_factor,
                patience=args.reduce_patience,
                cooldown=args.reduce_cooldown,
                min_lr=args.reduce_min_lr,
                threshold=args.reduce_lr_threshold,
                verbose=True)
        else:
            raise ValueError(f"Unsupported scheduler type: {args.lr_scheduler}")
            
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
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # 根据测试集表现保存最佳模型
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric_test))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric_test]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Test performance didn't improve for {} epochs. Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        # 仅关注测试集结果
        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][self.mnt_metric_test])
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
    # 定义 ANSI 颜色代码作为类的静态属性
    BLUE = "\033[34m"  # 蓝色
    YELLOW = "\033[33m"  # 新增黄色
    ENDC = "\033[0m"  # 结束颜色
    
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader, lr_ve, lr_ed, step_size, gamma):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.scheduler = build_lr_scheduler(args, optimizer)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_ve_history = []  # 记录Visual Extractor的学习率历史
        self.lr_ed_history = []  # 记录Encoder Decoder的学习率历史

        self.train_loss_history = []
        self.val_metrics_history = {}
        self.test_metrics_history = {}
        
        self.lr_ve = lr_ve  # 设置 lr_ve 属性
        self.lr_ed = lr_ed  # 设置 lr_ed 属性
        
        
    def translate_to_chinese(self, text_to_translate):
        # 创建一个Translator对象
        translator = Translator()
        try:
            # 尝试执行文本翻译，从英文到中文
            translated_text = translator.translate(text_to_translate, src='en', dest='zh-cn').text
            return translated_text
        except Exception as e:
            print(f"翻译时出现错误: {e}")
            # 在发生异常时返回原始文本或适当的错误消息
            return "翻译错误"  # 或者返回 text_to_translate
    
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

            if batch_idx % 2000 == 0:
                loss_message = '[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'.format(epoch, self.epochs, batch_idx, len(self.train_dataloader), train_loss / (batch_idx + 1))
                print(loss_message)
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        avg_train_loss = train_loss / len(self.train_dataloader)
        self.train_loss_history.append(avg_train_loss)  # 更新训练损失历史记录
        log = {'train_loss': avg_train_loss}
        print(f"[{epoch}/{self.epochs}] End of epoch average training loss: {log['train_loss']:.5f}. Epoch Time: {train_time:.2f} seconds")

        # 验证部分
        val_start_time = time.time()
        self.model.eval()
        val_gts, val_res = [], []
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

#                 if batch_idx == 0:
#                     random_indices = np.random.choice(len(images_id), 2, replace=False)
#                     for idx in random_indices:
#                         image_name = images_id[idx]
#                         translated_inference_text = self.translate_to_chinese(reports[idx])
#                         translated_ground_truth_text = self.translate_to_chinese(ground_truths[idx])
                        
#                         print(f"Validation Set - Image Name: {image_name}")
#                         print(f"\033[34mValidation Set - Inference Text: {reports[idx]} (Translated: {translated_inference_text})\033[0m")
#                         print(f"Validation Set - Ground Truth Text: {ground_truths[idx]} (Translated: {translated_ground_truth_text})")
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)}, {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            
            # composite_score = (val_met['BLEU_1'] + val_met['BLEU_2'] + val_met['BLEU_3'] + val_met['BLEU_4'] + val_met['METEOR'] + val_met['ROUGE_L']) / 6
            # self.scheduler.step(composite_score)
            # 在每个epoch结束时更新学习率
            bleu_4_score = val_met.get('BLEU_4', 0)  # 安全地获取BLEU_4分数，如果不存在则返回0

            # 在每个epoch结束时更新学习率
            if isinstance(self.scheduler, GradualWarmupScheduler):
                if self.scheduler.last_epoch < self.scheduler.total_epoch:
                    self.scheduler.step()
                else:
                    # 预热期结束，简单地步进后续调度器
                    self.scheduler.after_scheduler.step()
            else:
                self.scheduler.step()  # 对于StepLR和其他不依赖性能指标的调度器
    
            # 更新学习率历史记录
            current_lr_ve = self.optimizer.param_groups[0]['lr']
            current_lr_ed = self.optimizer.param_groups[1]['lr']
            self.lr_ve_history.append(current_lr_ve)
            self.lr_ed_history.append(current_lr_ed)
            print(f"{self.YELLOW}--lr_ve {current_lr_ve:.1e}{self.ENDC}")
            print(f"{self.YELLOW}--lr_ed {current_lr_ed:.1e}{self.ENDC}")

        val_end_time = time.time()
        val_time = val_end_time - val_start_time
        for metric_name, metric_value in val_met.items():
            if metric_name not in self.val_metrics_history:
                self.val_metrics_history[metric_name] = []
            self.val_metrics_history[metric_name].append(metric_value)
        
        test_start_time = time.time()
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

#                 if batch_idx == 0:
#                     random_indices = np.random.choice(len(images_id), 2, replace=False)
#                     for idx in random_indices:
#                         image_name = images_id[idx]
#                         translated_inference_text = self.translate_to_chinese(reports[idx])
#                         translated_ground_truth_text = self.translate_to_chinese(ground_truths[idx])

#                         print(f"Test Set - Image Name: {image_name}")
#                         print(f"\033[34mTest Set - Inference Text: {reports[idx]} (Translated: {translated_inference_text})\033[0m")
#                         print(f"Test Set - Ground Truth Text: {ground_truths[idx]} (Translated: {translated_ground_truth_text})")

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)}, {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        for metric_name, metric_value in test_met.items():
            if metric_name not in self.test_metrics_history:
                self.test_metrics_history[metric_name] = []
            self.test_metrics_history[metric_name].append(metric_value)

        if epoch % 10 == 0:
            title = f"Epoch: {epoch}, LR VE: {self.lr_ve}, LR ED: {self.lr_ed}"
            self._plot_metrics(epoch, title)

        print(f"[{epoch}/{self.epochs}] Train Time: {train_time:.2f} seconds")
        print(f"[{epoch}/{self.epochs}] Validation Time: {val_time:.2f} seconds")
        print(f"[{epoch}/{self.epochs}] Test Time: {test_time:.2f} seconds")

        return log
    
    def _plot_metrics(self, epoch, title=''):
        try:
            # 构建当前图表的文件路径
            current_image_path = f'/kaggle/working/metrics_epoch_{epoch}_{title}.png'

            # 查找并删除之前的图表文件
            for previous_image in os.listdir('/kaggle/working/'):
                if previous_image.startswith(f'metrics_epoch_') and previous_image.endswith('.png'):
                    os.remove(f'/kaggle/working/{previous_image}')

            # 创建图表
            plt.figure(figsize=(18, 22))  # 调整图表的总大小以适应新的绘图
            gs = gridspec.GridSpec(4, 1)  # 增加到4行以容纳学习率的绘图
            
            # 新增加的学习率绘制区域
            plt.subplot(gs[3])
            plt.plot(self.lr_ve_history, label='LR VE (Visual Extractor)', color='cyan', linestyle='--')
            plt.plot(self.lr_ed_history, label='LR ED (Encoder Decoder)', color='magenta', linestyle='--')
            plt.title('Learning Rate Changes', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Learning Rate', fontsize=15)
            plt.legend(fontsize=12)
            plt.grid(True)
        
            # 绘制训练损失并添加数值标注
            plt.subplot(gs[0])
            plt.plot(self.train_loss_history, label='Training Loss', color='blue')
            for x, y in enumerate(self.train_loss_history):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
            plt.title(f'Training Loss (Epoch {epoch})', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.legend(fontsize=12)
            plt.grid(True)

            # 绘制验证评价指标并添加数值标注
            plt.subplot(gs[1])
            avg_val_metrics = np.mean([v for k, v in self.val_metrics_history.items() if k != 'BLEU_4'], axis=0)
            plt.plot(avg_val_metrics, label='Average Validation Metrics', color='green')
            for x, y in enumerate(avg_val_metrics):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
            if 'BLEU_4' in self.val_metrics_history:
                plt.plot(self.val_metrics_history['BLEU_4'], label='BLEU_4', color='red')
                for x, y in enumerate(self.val_metrics_history['BLEU_4']):
                    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
            plt.title('Validation Metrics', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Metric Value', fontsize=15)
            plt.legend(fontsize=12)
            plt.grid(True)

            # 绘制测试评价指标并添加数值标注
            plt.subplot(gs[2])
            avg_test_metrics = np.mean([v for k, v in self.test_metrics_history.items() if k != 'BLEU_4'], axis=0)
            plt.plot(avg_test_metrics, label='Average Test Metrics', color='orange')
            for x, y in enumerate(avg_test_metrics):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
            if 'BLEU_4' in self.test_metrics_history:
                plt.plot(self.test_metrics_history['BLEU_4'], label='BLEU_4', color='purple')
                for x, y in enumerate(self.test_metrics_history['BLEU_4']):
                    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
            plt.title('Test Metrics', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Metric Value', fontsize=15)
            plt.legend(fontsize=12)
            plt.grid(True)

            # 使用lr_ve和lr_ed设置图表标题
            chart_title = f'LR VE: {self.lr_ve}, LR ED: {self.lr_ed}, {title}'
            plt.suptitle(chart_title, fontsize=24)
            plt.tight_layout()

            # 保存图表
            plt.savefig(current_image_path, dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error occurred during plotting: {e}")
        finally:
            plt.close()

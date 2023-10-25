import logging
import os
from abc import abstractmethod
import time
import torch
import random
import openai
from numpy import inf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .base_cmn import BaseCMN
from torch.cuda.amp import autocast, GradScaler
from googletrans import Translator

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

        # 先初始化为None
        self.train_loss_history = []
        self.val_metrics_history = None
        self.test_metrics_history = None
        
        self.lr_ve = lr_ve  # 设置 lr_ve 属性
        self.lr_ed = lr_ed  # 设置 lr_ed 属性
        self.step_size = step_size  # 设置 step_size 属性
        self.gamma = gamma  # 设置 gamma 属性
    
    def translate_to_chinese(self, text_to_translate):
        # 创建一个Translator对象
        translator = Translator()
        # 执行文本翻译，从英文到中文
        translated_text = translator.translate(text_to_translate, src='en', dest='zh-cn').text
        return translated_text
        
    def _train_epoch(self, epoch):
        print(logging.getLogger().getEffectiveLevel())
#         print('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        random_test_sample_idx = random.randint(0, len(self.test_dataloader) - 1)  # 移至循环外部
        random_val_sample_idx = random.randint(0, len(self.val_dataloader) - 1)  # 新增，移至循环外部
        self.model.train()
        
        start_time = time.time()  # 记录开始时间

        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            # 前向传播
            with autocast():
                output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)
            
            # 反向传播和梯度缩放
            train_loss += loss.item()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if batch_idx % self.args.log_period == 0:
                loss_message = '[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'.format(
                                epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                train_loss / (batch_idx + 1))
                print(loss_message)

        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time  # 计算时间

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        end_message = '[{}/{}] End of epoch average training loss: {:.5f}. Epoch Time: {:.2f} seconds'.format(
            epoch, self.epochs, log['train_loss'], epoch_time)
        print(end_message)
        
#         print('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        start_time = time.time()  # 记录开始时间

        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                
                # 添加随机验证样本的推理文本和原始文本
                if batch_idx == 0:  # 仅在第一个批次中打印图片信息
                    image_name = images_id[0]  # 第一个样本的图片名称
                    # 翻译文本
                    translated_inference_text = self.translate_to_chinese(reports[0])
                    translated_ground_truth_text = self.translate_to_chinese(ground_truths[0])
                    print("Image Name: ", image_name)
                    print("Inference Text: ", reports[0])
                    print("Inference Text (Translated):", translated_inference_text)
                    print("Ground Truth Text: ", ground_truths[0])
                    print("Ground Truth Text (Translated):", translated_ground_truth_text)
                    val_res.append(reports[0])
                    val_gts.append(ground_truths[0])

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        end_time = time.time()  # 记录结束时间
        val_time = end_time - start_time  # 计算时间

        if self.val_metrics_history is None:
                self.val_metrics_history = {k: [] for k in val_met}
        for k, v in val_met.items():
            self.val_metrics_history[k].append(v)

#         print('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()

        start_time = time.time()  # 记录开始时间

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                
                # 添加随机测试样本的推理文本和原始文本
                if batch_idx == 0:  # 仅在第一个批次中打印图片信息
                    image_name = images_id[0]  # 第一个样本的图片名称
                    # 翻译文本
                    translated_inference_text = self.translate_to_chinese(reports[0])
                    translated_ground_truth_text = self.translate_to_chinese(ground_truths[0])
                    print("Image Name: ", image_name)
                    print("Inference Text: ", reports[0])
                    print("Inference Text (Translated):", translated_inference_text)
                    print("Ground Truth Text: ", ground_truths[0])
                    print("Ground Truth Text (Translated):", translated_ground_truth_text)
                    test_res.append(reports[0])
                    test_gts.append(ground_truths[0])

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        end_time = time.time()  # 记录结束时间
        test_time = end_time - start_time  # 计算时间
        if self.test_metrics_history is None:
                self.test_metrics_history = {k: [] for k in test_met}
        for k, v in test_met.items():
            self.test_metrics_history[k].append(v)

        
        # 打印训练时间
        print(f"[{epoch}/{self.epochs}] Train Time: {epoch_time:.2f} seconds")
        # 打印验证时间
        print(f"[{epoch}/{self.epochs}] Validation Time: {val_time:.2f} seconds")
        # 打印测试时间
        print(f"[{epoch}/{self.epochs}] Test Time: {test_time:.2f} seconds")

        self.train_loss_history.append(log['train_loss'])
        self.lr_scheduler.step()
        if epoch % 10 == 0:
            self._plot_metrics(epoch)

        return log
    
    def _plot_metrics(self, epoch):
        # 从 self 对象中获取需要的参数
        lr_ve = self.lr_ve
        lr_ed = self.lr_ed
        step_size = self.step_size
        gamma = self.gamma

        # 构造包含参数的图片名称
        image_name = f've_{lr_ve}_ed_{lr_ed}_step_{step_size}_gamma_{gamma}_epoch_{epoch}.png'

        # 创建一个图表，包含三个子图
        plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2)

        # 绘制训练损失
        plt.subplot(gs[0, :])
        plt.plot(self.train_loss_history, label='Training Loss', color='blue')
        plt.title(f'Training Loss (Epoch {epoch})')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制验证指标
        if self.val_metrics_history is not None:
            plt.subplot(gs[1, 0])
            colors = ['green', 'orange', 'red', 'purple', 'brown', 'pink']
            for i, (metric_name, metric_values) in enumerate(self.val_metrics_history.items()):
                color = colors[i % len(colors)]
                plt.plot(metric_values, label=f'Validation {metric_name}', color=color)
            plt.title(f'Validation Metrics (Epoch {epoch})')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()

        # 绘制测试指标
        if self.test_metrics_history is not None:
            plt.subplot(gs[1, 1])
            colors = ['green', 'orange', 'red', 'purple', 'brown', 'pink']
            for i, (metric_name, metric_values) in enumerate(self.test_metrics_history.items()):
                color = colors[i % len(colors)]
                plt.plot(metric_values, label=f'Test {metric_name}', color=color)
            plt.title(f'Test Metrics (Epoch {epoch})')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()

        plt.tight_layout()

        # 保存图表为图片文件
        image_path = os.path.join('/kaggle/working/', image_name)  # 替换为你想要保存图像的目录
        plt.savefig(image_path, dpi=300)

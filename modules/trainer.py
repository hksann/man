import logging
import os
from abc import abstractmethod

import torch
from numpy import inf
import matplotlib.pyplot as plt

from .base_cmn import BaseCMN


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
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
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
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
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
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There's no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            msg = "Warning: The number of GPU's configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
            self.logger.warning(msg)
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
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
         # 先初始化为None
        self.train_loss_history = []
        self.val_metrics_history = None
        self.test_metrics_history = None
        
        
    def _train_epoch(self, epoch):
        self.logger.info(logging.getLogger().getEffectiveLevel())
        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()

        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            
#             print(f"Device for images: {images.device}")
#             print(f"Device for reports_ids: {reports_ids.device}")
#             print(f"Device for reports_masks: {reports_masks.device}")
            
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
#             images = images.to(self.device)
#             reports_ids = reports_ids.to(self.device)
#             reports_masks = reports_masks.to(self.device)

            output = self.model(images, reports_ids, mode='train')

            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_period == 0:
                loss_message = '[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'.format(
                                epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                train_loss / (batch_idx + 1))
                self.logger.info(loss_message)
#                 print(loss_message)  # 打印到标准输出

        # 使用您提供的方式来计算平均训练损失并更新log字典
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        end_message = '[{}/{}] End of epoch average training loss: {:.5f}.'.format(epoch, self.epochs, log['train_loss'])
        self.logger.info(end_message)
        print(end_message)  # 打印到标准输出
        
        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()

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

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

            print(f"Epoch {epoch} - Validation Metrics:")
            for k, v in val_met.items():
                print(f"{k}: {v}")
            print()

            if self.val_metrics_history is None:
                self.val_metrics_history = {k: [] for k in val_met}
            for k, v in val_met.items():
                self.val_metrics_history[k].append(v)

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()

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

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

            print(f"Epoch {epoch} - Test Metrics:")
            for k, v in test_met.items():
                print(f"{k}: {v}")
            print()

            if self.test_metrics_history is None:
                self.test_metrics_history = {k: [] for k in test_met}
            for k, v in test_met.items():
                self.test_metrics_history[k].append(v)
        
        self.train_loss_history.append(log['train_loss'])
        self.lr_scheduler.step()
        self._plot_metrics()
        
        print(log)
        return log


    def _plot_metrics(self):
        # num_metrics现在包括了train_loss，所以要加1
        num_metrics = len(self.val_metrics_history) + 1

        # 创建一个新的图形窗口
        fig, axs = plt.subplots(num_metrics, 1, figsize=(12, 5*num_metrics))

        # 检查保存目录是否存在，不存在则创建
        save_dir = '/kaggle/working/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # 先绘制train_loss
        epochs = range(1, len(self.train_loss_history) + 1)
        axs[0].plot(epochs, self.train_loss_history, label='Train Loss')
        axs[0].set_title("Train Loss over epochs")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        # 绘制其他指标
        for idx, metric_name in enumerate(self.val_metrics_history, start=1):  # 从1开始，因为第0个已经被train_loss使用了
            epochs = range(1, len(self.val_metrics_history[metric_name]) + 1)

            axs[idx].plot(epochs, self.val_metrics_history[metric_name], label=f'Validation {metric_name}')
            axs[idx].plot(epochs, self.test_metrics_history[metric_name], label=f'Test {metric_name}', linestyle='--')

            axs[idx].set_title(f"{metric_name} over epochs")
            axs[idx].set_xlabel("Epochs")
            axs[idx].set_ylabel(metric_name)
            axs[idx].legend()
            axs[idx].grid(True)

        # 如果当前是第5个epoch或者是最后一个epoch，则保存图片
        current_epoch = len(epochs)
        if current_epoch % 5 == 0 or current_epoch == max(epochs):
            # 删除save_dir目录下旧的图片
            for old_image in os.listdir(save_dir):
                if "combined_metrics" in old_image:
                    os.remove(os.path.join(save_dir, old_image))

            plt.savefig(os.path.join(save_dir, f"combined_metrics_epoch_{current_epoch}.png"),dpi=300)

        # 直接显示图形
        plt.show()

        # 关闭当前图形释放资源
        plt.close(fig)
        # 清除当前图形，为下一个指标做准备
        plt.clf()

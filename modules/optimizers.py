import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR

class GradualWarmupScheduler(_LRScheduler):
    """Gradual Warmup Scheduler."""
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        # Ensure last_epoch starts at -1 so that the scheduler updates correctly
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        # Skip warm-up if total_epoch is 0
        if self.total_epoch == 0:
            if self.after_scheduler and hasattr(self.after_scheduler, 'get_lr'):
                return self.after_scheduler.get_lr()
            else:
                return [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch - 1 < self.total_epoch:
            return [(base_lr * ((self.multiplier - 1) * self.last_epoch / self.total_epoch + 1)) for base_lr in self.base_lrs]
        else:
            if self.after_scheduler and hasattr(self.after_scheduler, 'get_lr'):
                return self.after_scheduler.get_lr()
            else:
                return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        if self.total_epoch == 0:
            if self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch)
        elif self.last_epoch + 1 > self.total_epoch:
            if self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch if epoch is not None else None)
        else:
            return super().step(epoch)

def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda p: id(p) not in ve_params, model.parameters())

    OptimizerClass = optim.Adam if args.optim.lower() == 'adam' else optim.AdamW
    optimizer = OptimizerClass(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay
    )
    return optimizer

def build_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'ReduceLROnPlateau':
        after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.reduce_factor, patience=args.reduce_patience, verbose=True)
    elif args.lr_scheduler == 'StepLR':
        after_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        after_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)

    scheduler = GradualWarmupScheduler(optimizer, multiplier=args.multiplier, total_epoch=args.warmup_epochs, after_scheduler=after_scheduler)
    return scheduler

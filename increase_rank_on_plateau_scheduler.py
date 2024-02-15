# Based on PyTorch's ReduceLROnPlateau scheduler (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
from torch import inf


class IncreaseRankOnPlateau:
    def __init__(self, n_iters, end_rank, start_rank=1, factor=2, patience=15, factor_type='mult',
                 force_end_rank_percent=0.5, threshold=0, threshold_mode='rel', cooldown=0, mode='min', verbose=False, logger=None):
        self.n_iters = n_iters
        self.force_end_rank_percent = force_end_rank_percent
        self.force_end_rank_step = n_iters - (n_iters * force_end_rank_percent)
        self.factor_type = factor_type
        self.factor = factor
        self.start_rank = start_rank
        self.curr_rank = start_rank
        self.end_rank = end_rank
        self.logger = logger

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor

        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch < self.force_end_rank_step:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._increase_rank(epoch, current)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
        else:
            old_rank = self.curr_rank
            new_rank = self.end_rank
            self.curr_rank = new_rank
            if new_rank > old_rank and self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                if self.logger is not None:
                    self.logger.info(f'Epoch {epoch_str}: forcing end rank (end_rank={new_rank}). Current loss={current}')
                else:
                    print(f'Epoch {epoch_str}: forcing end rank (end_rank={new_rank}). Current loss={current}')

    def _increase_rank(self, epoch, loss):
        old_rank = self.curr_rank
        if self.factor_type == "mult":
            new_rank = min(old_rank * self.factor, self.end_rank)
        elif self.factor_type == "add":
            new_rank = min(old_rank + 1, self.end_rank)
        self.curr_rank = new_rank
        if new_rank > old_rank and self.verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            if self.logger is not None:
                self.logger.info(f'Epoch {epoch_str}: increasing rank to {new_rank}. Current loss={loss}')
            else:
                print(f'Epoch {epoch_str}: increasing rank to {new_rank}. Current loss={loss}')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

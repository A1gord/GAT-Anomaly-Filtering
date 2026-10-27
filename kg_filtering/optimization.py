import torch


def build_optimizer(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
        factor=config.scheduler_factor, patience=config.scheduler_patience)
    return optimizer, scheduler


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.score = float('-inf')
        self.epoch = 0
        self.stale = 0
        self.state = None

    def update(self, score, model, epoch):
        if score > self.score:
            self.score, self.epoch, self.stale = score, epoch, 0
            self.state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            self.stale += 1
        return self.stale >= self.patience

    def restore(self, model):
        if self.state is None:
            raise RuntimeError('No model state was selected')
        model.load_state_dict(self.state)

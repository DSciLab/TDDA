from mlutils import Trainer


class BaseTrainer(Trainer):
    def __init__(self, opt):
        super().__init__(opt)

    def train_step(self, item):
        # return loss, preds, labels, ....
        raise NotImplementedError

    def eval_step(self, item):
        # return loss, preds, labels, ....
        raise NotImplementedError

    def inference(self, image):
        raise NotImplementedError

    def on_epoch_begin(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def on_training_begin(self):
        raise NotImplementedError

    def on_training_end(self):
        raise NotImplementedError
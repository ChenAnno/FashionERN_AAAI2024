import torch


class BaseTrainer:
    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        self.local_rank = kwargs["local_rank"]
        self.num_epochs = kwargs["num_epochs"]
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.validation_frequency = kwargs["validation_frequency"]
        self.print_frequency = kwargs["print_frequency"]
        self.clip_model_name = kwargs["clip_model_name"]
        self.feature_dim = kwargs["feature_dim"]
        self.target_ratio = kwargs["target_ratio"]
        self.input_dim = kwargs["input_dim"]
        self.num_workers = kwargs["num_workers"]
        self.patch_num = kwargs["patch_num"]
        self.clip_bs = kwargs["clip_bs"]
        self.dataset = kwargs["dataset"]

        self.model, self.tokenizer = self.define_model()

        self.train_loader = self.define_train_loader()

        self.optimizer, self.scheduler, self.criterion, self.scaler = self.define_optimizer_and_loss()

        self.epoch = 0

    def define_model(self):
        raise NotImplementedError("Subclasses should implement {}".format("define_model"))

    def define_train_loader(self):
        raise NotImplementedError("Subclasses should implement {}".format("define_train_loader"))

    def define_val_datasets(self):
        raise NotImplementedError("Subclasses should implement {}".format("define_val_datasets"))

    def define_optimizer_and_loss(self):
        raise NotImplementedError("Subclasses should implement {}".format("define_optimizer_and_loss"))

    def train_one_epoch(self):
        raise NotImplementedError("Subclasses should implement {}".format("train_one_epoch"))

    def validate(self):
        if self.dataset != "fashion200k":
            raise NotImplementedError("Subclasses should implement {}".format("validate"))

    def train(self):
        self.model.train()
        for self.epoch in range(self.num_epochs):
            self.train_one_epoch()
            if self.epoch % self.validation_frequency == 0:
                if self.dataset != "fashion200k":
                    self.validate()

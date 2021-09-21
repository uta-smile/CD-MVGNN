import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dglt.contrib.grover.models import GroverTask
from dglt.contrib.grover.models import MolEmbedding
from dglt.contrib.grover.molbertcollator import GroverCollator
from dglt.contrib.grover.vocab import AtomVocab
from dglt.data.dataset.utils import get_data, split_data, get_task_names
from dglt.models.nn_utils import param_count
from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw
from dglt.utils import build_optimizer, build_lr_scheduler


class MolBERTTrainer:
    def __init__(self,
                 args,
                 molbert,
                 vocab_size,
                 fg_szie,
                 train_dataloader,
                 test_dataloader,
                 optimizer_builder,
                 scheduler_builder,
                 logger=None,
                 with_cuda=False,
                 enable_multi_gpu=False):

        self.args = args
        self.with_cuda = with_cuda
        self.molbert = molbert
        self.model = GroverTask(args, molbert, vocab_size, fg_szie)
        self.loss_func = self.model.get_loss_func(args)

        self.vocab_size = vocab_size
        self.debug = logger.debug if logger is not None else print

        if self.with_cuda:
            # print("Using %d GPUs for training." % (torch.cuda.device_count()))
            self.model = self.model.cuda()


        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optimizer = optimizer_builder(self.model, self.args)
        self.scheduler = scheduler_builder(self.optimizer, self.args)
        if enable_multi_gpu:
            # broadcast parameters & optimizer state.
            mgw.broadcast_parameters(self.model.state_dict(), root_rank=0)
            self.optimizer = mgw.DistributedOptimizer(self.optimizer,
                                                      named_parameters=self.model.named_parameters())


        self.args = args
        self.n_iter = 0

    def train(self, epoch):
        # return self.mock_iter(epoch, self.train_data, train=True)
        return self.iter(epoch, self.train_data, train=True)

    def test(self, epoch):
        #return self.mock_iter(epoch, self.test_data, train=False)
        return self.iter(epoch, self.test_data, train=False)

    def mock_iter(self, epoch, data_loader, train=True):

        for i, item in enumerate(data_loader):
            self.scheduler.step()
        cum_loss_sum = 0.0
        av_loss_sum, fg_loss_sum, av_dist_loss_sum, fg_dist_loss_sum = 0, 0, 0, 0
        self.n_iter += self.args.batch_size
        return self.n_iter, cum_loss_sum, (av_loss_sum, fg_loss_sum, av_dist_loss_sum, fg_dist_loss_sum)

    def iter(self, epoch, data_loader, train=True):

        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_sum, iter_count = 0, 0
        cum_loss_sum, cum_iter_count = 0, 0
        av_loss_sum, fg_loss_sum, av_dist_loss_sum, fg_dist_loss_sum = 0, 0, 0, 0
        #loss_func = self.model.get_loss_func(self.args)

        for i, item in enumerate(data_loader):
            batch_graph = item["graph_input"]
            targets = item["targets"]
            # print(type(batch_graph))
            if next(self.model.parameters()).is_cuda:
                targets["av_task"] = targets["av_task"].cuda()
                targets["fg_task"] = targets["fg_task"].cuda()

            preds = self.model(batch_graph)
            loss, av_loss, fg_loss, av_dist_loss, fg_dist_loss = self.loss_func(preds, targets)

            loss_sum += loss.item()
            iter_count += self.args.batch_size

            if train:
                cum_loss_sum += loss.item()
                # Run model
                self.model.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
                # For eval model, only consider the loss of two task.
                cum_loss_sum += av_loss.item()
                cum_loss_sum += fg_loss.item()

            cum_iter_count += 1
            self.n_iter += self.args.batch_size

        cum_loss_sum /= cum_iter_count
        av_loss_sum /= cum_iter_count
        fg_loss_sum /= cum_iter_count
        av_dist_loss_sum /= cum_iter_count
        fg_dist_loss_sum /= cum_iter_count

        return self.n_iter, cum_loss_sum, (av_loss_sum, fg_loss_sum, av_dist_loss_sum, fg_dist_loss_sum)

    def save(self, epoch, file_path):

        output_path = file_path + ".ep%d" % epoch
        scaler = None
        features_scaler = None
        state = {
            'args': self.args,
            'state_dict': self.model.cpu().state_dict(),
            'data_scaler': {
                'means': scaler.means,
                'stds': scaler.stds
            } if scaler is not None else None,
            'features_scaler': {
                'means': features_scaler.means,
                'stds': features_scaler.stds
            } if features_scaler is not None else None
        }
        torch.save(state, output_path)

        # Is this one necessary?
        if self.with_cuda:
            self.model = self.model.cuda()
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


def run_training(args, logger):
    # initialize library
    if args.enable_multi_gpu:
        mgw.init()

    master_worker = (mgw.rank() == 0) if args.enable_multi_gpu else True
    # pin GPU to local rank.
    idx = mgw.local_rank() if args.enable_multi_gpu else args.gpu

    # if args.gpu is not None:
    torch.cuda.set_device(idx)

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    if master_worker:
        print(args)
        if args.enable_multi_gpu:
            debug("Total workers: %d" % (mgw.size()))
        debug('Loading data')

    args.task_names = get_task_names(args.data_path)

    data = get_data(path=args.data_path,
                    features_path=args.fg_label_path,
                    args=args, logger=logger)
    if master_worker:
        debug(f'Splitting data with seed {args.seed}')
    train_data, test_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                          seed=args.seed, args=args, logger=logger)

    # Here the true train data size is the train_data divided by #GPUs
    if args.enable_multi_gpu:
        args.train_data_size = len(train_data) // mgw.size()
    else:
        args.train_data_size = len(train_data)
    if master_worker:
        debug(f'Total size = {len(data):,} | '
              f'train size = {len(train_data):,} | val size = {len(test_data):}')

    vocab = AtomVocab.load_vocab(args.vocab_path)
    vocab_size = len(vocab)
    fg_size = train_data.data[0].features.shape[0]
    if master_worker:
        debug("Vocab size: %d, Number of FG tasks: %d" % (vocab_size, fg_size))

    shared_dict = {}
    mol_collator = GroverCollator(shared_dict=shared_dict, vocab=vocab, args=args)
    train_sampler = None
    test_sampler = None
    shuffle = True
    if args.enable_multi_gpu:
        train_sampler = DistributedSampler(
            train_data, num_replicas=mgw.size(), rank=mgw.rank())
        test_sampler = DistributedSampler(
            test_data, num_replicas=mgw.size(), rank=mgw.rank())
        train_sampler.set_epoch(args.epochs)
        test_sampler.set_epoch(1)
        shuffle = False

    train_data = DataLoader(train_data,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            num_workers=10,
                            sampler=train_sampler,
                            collate_fn=mol_collator)
    test_data = DataLoader(test_data,
                           batch_size=args.batch_size,
                           shuffle=shuffle,
                           num_workers=10,
                           sampler=test_sampler,
                           collate_fn=mol_collator)

    molbert = MolEmbedding(args)

    trainer = MolBERTTrainer(args=args,
                             molbert=molbert,
                             vocab_size=vocab_size,
                             fg_szie=fg_size,
                             train_dataloader=train_data,
                             test_dataloader=test_data,
                             optimizer_builder=build_optimizer,
                             scheduler_builder=build_lr_scheduler,
                             logger=logger,
                             with_cuda=True,
                             enable_multi_gpu=args.enable_multi_gpu)

    if master_worker:
        print("Total parameters: %d" % param_count(trainer.molbert))

    # model_dir = os.path.join(args.save_dir, "models")
    # if master_worker:
    #    if not os.path.exists(model_dir):
    #        os.mkdir(model_dir)
    model_dir = os.path.join(args.save_dir, "model")
    min_val_loss = float('inf')
    for epoch in range(args.epochs):
        s_time = time.time()
        n_iter, train_loss, detailed_loss = trainer.train(epoch)
        t_time = time.time() - s_time
        s_time = time.time()
        _, val_loss, _ = trainer.test(epoch)
        v_time = time.time() - s_time
        if master_worker:
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(val_loss),
                  # f'{args.metric}_val: {avg_val_score:.4f}',
                  # 'auc_val: {:.4f}'.format(avg_val_score),
                  'cur_lr: {:.5f}'.format(trainer.scheduler.get_lr()[0]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time))

            if epoch % args.save_interval == 0 and val_loss < min_val_loss:
                trainer.save(epoch, model_dir)
                min_val_loss = val_loss
    # Only save final version.
    if master_worker:
        trainer.save(args.epochs, model_dir)

from ._utils import _weights_path
from .graph import LiveGraph

import torch
import time
import logging
import os, re
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from datetime import date
"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""


class Trainer(object):
    def __init__(self, model, loss_func, optimizer, scheduler=None, gpu=True, log_interval=100):
        self.gpu = gpu

        self.model = model.cuda() if self.gpu else model
        # convert to float
        self.model = self.model.to(dtype=torch.float)
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.scheduler = scheduler

        self.test_losses = []

        self.log_interval = log_interval

    """
    @property
    def model_name(self):
        return self.model.__class__.__name__.lower()
    """

    def train(self, epochs, train_loader, savemodelname='tracknet', checkpoints_epoch_interval=50, max_checkpoints=15, live_graph=None):
        """
        :param epochs: int, how many iterations during training
        :param train_loader: Dataloader, must return Tensor of images and ground truthes
        :param savemodelname: (Optional) str or None, saved model name. if it's None, model will not be saved after finishing training.
        :param checkpoints_epoch_interval: (Optional) int or None, Whether to save for each designated iteration or not. if it's None, model will not be saved.
        :param max_checkpoints: (Optional) int, how many models will be saved during training.
        :return:
        """
        self.model.train()

        log_manager = _LogManager(savemodelname, checkpoints_epoch_interval, max_checkpoints, epochs, live_graph)

        for epoch in range(1, epochs + 1):
            for _iteration, (images, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()

                if self.gpu:
                    images = images.cuda()
                    targets = targets.cuda()

                # set variable
                #images.requires_grad = True
                #gts.requires_grad = True

                predicts = self.model(images)

                loss = self.loss_func(predicts, targets)
                loss.backward() # calculate gradient for value with requires_grad=True, shortly back propagation
                #print(self.model.feature_layers.conv1_1.weight.grad)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
            
                # update log
                log_manager.update_log(epoch, _iteration + 1, batch_num=len(images), data_num=len(train_loader.dataset),
                                       iter_per_epoch=len(train_loader), lossval=loss.item())
                log_manager.store_iter_loss(lossval=loss.item())
            # update info
            log_manager.update_log_epoch(epoch)

            # save checkpoints
            log_manager.save_checkpoints_model(epoch, self.model)

        print('Training finished')
        log_manager.save_model(self.model)
            
class _LogManager(object):
    def __init__(self, savemodelname, checkpoints_epoch_interval, max_checkpoints, epochs, live_graph):
        if savemodelname is None:
            logging.warning('Training model will not be saved!!!')

        if checkpoints_epoch_interval and max_checkpoints > 15:
            logging.warning('One model size will be about 0.1 GB. Please take care your storage.')

        savedir = _weights_path(__file__, _root_num=2, dirname='weights')
        save_checkpoints_dir = os.path.join(savedir, 'checkpoints')
        today = '{:%Y%m%d}'.format(date.today())

        # check existing checkpoints file
        filepaths = sorted(
            glob(os.path.join(save_checkpoints_dir, savemodelname + '_e[-]*_checkpoints{}.pth'.format(today))))
        if len(filepaths) > 0:
            logging.warning('Today\'s checkpoints is remaining. Remove them?\nInput any key. [n]/y')
            i = input()
            if re.match(r'y|yes', i, flags=re.IGNORECASE):
                for file in filepaths:
                    os.remove(file)
                logging.warning('Removed {}'.format(filepaths))
            else:
                logging.warning('Please rename them.')
                exit()

        if live_graph:
            logging.info("You should use jupyter notebook")
            if not isinstance(live_graph, LiveGraph):
                raise ValueError('live_graph must inherit LivaGraph')

            # initialise the graph and settings
            live_graph.initialize()

        # log's info
        self.savedir = savedir
        self.save_checkpoints_dir = save_checkpoints_dir
        self.savemodelname = savemodelname
        self.today = today

        # parameters
        self.checkpoints_epoch_interval = checkpoints_epoch_interval
        self.max_checkpoints = max_checkpoints
        self.epochs = epochs
        self.live_graph = live_graph

        self.train_losses = []
        self.train_losses_iteration = []
        self.total_iteration = 0


    def update_log(self, epoch, iteration, batch_num,
                   data_num, iter_per_epoch, lossval):
        #template = 'Epoch {}, Loss: {:.5f}, Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, elapsed_time {:.5f}'
        iter_template = 'Training... Epoch: {}, Iter: {},\t [{}/{}\t ({:.0f}%)]\tLoss: {:.6f}'
        """
        self.total_iteration += 1
        self.train_losses.append(lossval)
        self.train_losses_iteration.append(self.total_iteration)

        if self.live_graph:
            self.live_graph.redraw(epoch, iteration, self.train_losses_iteration, self.train_losses)
        else:
            print(iter_template.format(
                epoch, iteration, iteration * batch_num, data_num,
                                  100. * iteration / iter_per_epoch, lossval))
        """
        print(iter_template.format(
            epoch, iteration, iteration * batch_num, data_num,
                              100. * iteration / iter_per_epoch, lossval))

    def store_iter_loss(self, lossval):
        self.total_iteration += 1
        self.train_losses_iteration += [lossval]

    def update_log_epoch(self, epoch):
        self.train_losses += [np.mean(self.train_losses_iteration)]
        self.train_losses_iteration = []

        if self.live_graph:
            self.live_graph.redraw(epoch, self.total_iteration, np.arange(1, epoch + 1), self.train_losses)
        else:
            iter_template = 'Training... Epoch: {}, Iter: {},\tLoss: {:.6f}'
            print(iter_template.format(
                epoch, self.total_iteration, self.train_losses[-1]))

    def save_checkpoints_model(self, epoch, model):
        info = ''
        if epoch % self.checkpoints_epoch_interval == 0 and self.savemodelname and epoch != self.epochs:
            filepaths = sorted(
                glob(os.path.join(self.save_checkpoints_dir,
                                  self.savemodelname + '_e[-]*_checkpoints{}.pth'.format(self.today))))

            # filepaths = [path for path in os.listdir(save_checkpoints_dir) if re.search(savemodelname + '_i\-*_checkpoints{}.pth'.format(today), path)]
            # print(filepaths)
            removedinfo = ''
            # remove oldest checkpoints
            if len(filepaths) > self.max_checkpoints - 1:
                removedinfo += os.path.basename(filepaths[0])
                os.remove(filepaths[0])


            # save model
            savepath = os.path.join(self.save_checkpoints_dir,
                                    self.savemodelname + '_e-{:07d}_checkpoints{}.pth'.format(epoch, self.today))
            torch.save(model.state_dict(), savepath)

            # append information for verbose
            info += 'Saved model to {}'.format(savepath)
            if removedinfo != '':
                if self.live_graph:
                    removedinfo = '\nRemoved {}'.format(removedinfo)
                    info = '\n' + 'Saved model as {}{}'.format(os.path.basename(savepath), removedinfo)
                    self.live_graph.update_info(info)
                else:
                    removedinfo = ' and removed {}'.format(removedinfo)
                    info = '\n' + 'Saved model as {}{}'.format(os.path.basename(savepath), removedinfo)
                    print(info)

    def save_model(self, model):
        if self.savemodelname:
            # model
            savepath = os.path.join(self.savedir, self.savemodelname + '_e-{}.pth'.format(self.epochs))
            torch.save(model.state_dict(), savepath)
            print('Saved model to {}'.format(savepath))


            # graph
            savepath = os.path.join(self.savedir, self.savemodelname + '_learning-curve_e-{}.png'.format(self.epochs))
            # initialise the graph and settings
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            ax.clear()
            # plot
            ax.plot(self.train_losses_iteration, self.train_losses)
            ax.set_title('Learning curve')
            ax.set_xlabel('iteration')
            ax.set_ylabel('loss')
            #ax.axis(xmin=1, xmax=iterations)
            # save
            fig.savefig(savepath)

            print('Saved graph to {}'.format(savepath))
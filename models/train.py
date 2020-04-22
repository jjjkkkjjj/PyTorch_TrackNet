from ._utils import _weights_path

import torch
import time
import logging
import os, re
import matplotlib.pyplot as plt
import math
from glob import glob
from datetime import date
"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""


class Trainer(object):
    def __init__(self, model, loss_func, optimizer, scheduler=None, gpu=True, log_interval=100, live_graph=False, plot_yrange=(0, 0.16)):
        self.gpu = gpu

        self.model = model.cuda() if self.gpu else model
        # convert to float
        self.model = self.model.to(dtype=torch.float)
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.scheduler = scheduler

        self.live_graph = live_graph
        if self.live_graph:
            logging.info("You should use jupyter notebook")
        self._plot_yrange = plot_yrange

        self.train_losses = []
        self.train_losses_epoch = []
        self.test_losses = []

        self.log_interval = log_interval

    """
    @property
    def model_name(self):
        return self.model.__class__.__name__.lower()
    """

    def train(self, epochs, train_loader, savemodelname='tracknet', checkpoints_interval=50, max_checkpoints=15):
        """
        :param iterations: int, how many iterations during training
        :param train_loader: Dataloader, must return Tensor of images and ground truthes
        :param savemodelname: (Optional) str or None, saved model name. if it's None, model will not be saved after finishing training.
        :param checkpoints_interval: (Optional) int or None, Whether to save for each designated iteration or not. if it's None, model will not be saved.
        :param max_checkpoints: (Optional) int, how many models will be saved during training.
        :return:
        """
        if savemodelname is None:
            logging.warning('Training model will not be saved!!!')

        if checkpoints_interval and max_checkpoints > 15:
            logging.warning('One model size will be about 0.1 GB. Please take care your storage.')

        self.model.train()

        savedir = _weights_path(__file__, _root_num=2, dirname='weights')
        save_checkpoints_dir = os.path.join(savedir, 'checkpoints')
        today = '{:%Y%m%d}'.format(date.today())

        # check existing checkpoints file
        filepaths = sorted(glob(os.path.join(save_checkpoints_dir, savemodelname + '_e[-]*_checkpoints{}.pth'.format(today))))
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

        if self.live_graph:
            # initialise the graph and settings
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()

            fig.show()
            fig.canvas.draw()


        template = 'Epoch {}, Loss: {:.5f}, Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, elapsed_time {:.5f}'
        iter_template = 'Training... Epoch: {}, Iter: {},\t [{}/{}\t ({:.0f}%)]\tLoss: {:.6f}'
        _finish = False
        for epoch in range(1, epochs + 1):
            for _iteration, (images, gts) in enumerate(train_loader):
                now_iter = _iteration + 1

                self.optimizer.zero_grad()

                if self.gpu:
                    images = images.cuda()
                    gts = gts.cuda()

                # set variable
                #images.requires_grad = True
                #gts.requires_grad = True

                predicts, dboxes = self.model(images)
                if self.gpu:
                    dboxes = dboxes.cuda()
                loss = self.loss_func(predicts, gts, dboxes=dboxes)
                loss.backward() # calculate gradient for value with requires_grad=True, shortly back propagation
                #print(self.model.feature_layers.conv1_1.weight.grad)

                self.optimizer.step()
                self.scheduler.step()
            continue
            # save checkpoints
            appx_info = ''
            if epoch % checkpoints_interval == 0 and savemodelname and epoch != epochs:
                filepaths = sorted(glob(os.path.join(save_checkpoints_dir, savemodelname + '_e[-]*_checkpoints{}.pth'.format(today))))

                #filepaths = [path for path in os.listdir(save_checkpoints_dir) if re.search(savemodelname + '_i\-*_checkpoints{}.pth'.format(today), path)]
                #print(filepaths)
                removedinfo = ''
                # remove oldest checkpoints
                if len(filepaths) > max_checkpoints - 1:
                    removedinfo += os.path.basename(filepaths[0])
                    os.remove(filepaths[0])

                savepath = os.path.join(save_checkpoints_dir, savemodelname + '_e-{:07d}_checkpoints{}.pth'.format(epoch, today))
                torch.save(self.model.state_dict(), savepath)

                # append information for verbose
                appx_info += 'Saved model to {}'.format(savepath)
                if removedinfo != '':
                    if self.live_graph:
                        removedinfo = '\nRemoved {}'.format(removedinfo)
                    else:
                        removedinfo = ' and removed {}'.format(removedinfo)

                appx_info = '\n' + 'Saved model as {}{}'.format(os.path.basename(savepath), removedinfo)

            #print([param_group['lr'] for param_group in self.optimizer.param_groups])
            # store log
            self.train_losses.append(loss.item())
            self.train_losses_epoch.append(epoch)

            # update information to show
            if self.live_graph:
                ax.clear()
                # plot
                ax.plot(self.train_losses_epoch, self.train_losses)
                # ax.axis(xmin=0, xmax=iterations) # too small to see!!
                if self._plot_yrange:
                    ax.axis(ymin=self._plot_yrange[0], ymax=self._plot_yrange[1])
                if appx_info == '':
                    ax.set_title('Learning curve\nEpoch: {}, Iteration: {}, Loss: {}'.format(epoch, total_iteration,
                                                                                             loss.item()) + appx_info)
                else:
                    ax.set_title('Learning curve\nEpoch: {}, Iteration: {}, Loss: {}'.format(epoch, total_iteration,
                                                                                             loss.item()) + appx_info,
                                 fontsize=8)
                ax.set_xlabel('iteration')
                ax.set_ylabel('loss')
                # update
                fig.canvas.draw()

                """
                # not showing
                print(iter_template.format(
                    epoch, now_iter, now_iter * len(images), len(train_loader.dataset),
                           100. * now_iter / len(train_loader), loss.item()))
                """

            print(iter_template.format(
                        epoch, now_iter, now_iter * len(images), len(train_loader.dataset),
                               100. * now_iter / len(train_loader), loss.item()) + appx_info)

            #elif not self.live_graph and appx_info != '':
            #    print(appx_info[1:])



            """
            for test_image, test_label in zip(test_images, test_labels):
                self._test_step(test_image, test_label)

            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_acc.result() * 100,
                                  self.test_loss.result(),
                                  self.test_acc.result() * 100,
                                  elapsed_time))
            """
        print('Training finished')
        if savemodelname:
            savepath = os.path.join(savedir, savemodelname + '_i-{}.pth'.format(iterations))
            torch.save(self.model.state_dict(), savepath)
            print('Saved model to {}'.format(savepath))

            savepath = os.path.join(savedir, savemodelname + '_learning-curve_i-{}.png'.format(iterations))
            # initialise the graph and settings
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            ax.clear()
            # plot
            ax.plot(self.train_losses_iter, self.train_losses)
            ax.set_title('Learning curve')
            ax.set_xlabel('iteration')
            ax.set_ylabel('loss')
            #ax.axis(xmin=1, xmax=iterations)
            # save
            fig.savefig(savepath)

            print('Saved graph to {}'.format(savepath))
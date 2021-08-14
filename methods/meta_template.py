import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from abc import abstractmethod
import tqdm
import torch.nn.functional as F


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = (
            change_way
        )  # some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(
                self.n_way * (self.n_support + self.n_query), *x.size()[2:]
            )
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, : self.n_support]
        z_query = z_all[:, self.n_support :]

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query), scores

    def train_loop(self, epoch, train_loader, optimizer):
        ###################
        # optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=1e-3)

        avg_loss = 0
        pbar = tqdm.tqdm(train_loader)

        # print("***********",train_loader[0])############
        i = 0
        for x, _ in pbar:
            i += 1
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            pbar.set_description(
                "Epoch {:d} | Loss {:f}".format(epoch, avg_loss / float(i + 1))
            )

        print("a: {:.2f}  out scale: {:.5f}  length scale: {:.2f}  noise: {:.5f}".format(
                self.a.item(),
                F.softplus(self.kernel.output_scale_raw).item(),
                torch.exp(self.kernel.lengthscale_raw).item() if hasattr(self.kernel, 'lengthscale_raw') else float('nan'),
                F.softplus(self.noise).item()
            ))


        return avg_loss / i, self.a.item(), F.softplus(self.kernel.output_scale_raw).item(), torch.exp(self.kernel.lengthscale_raw).item() if hasattr(self.kernel, 'lengthscale_raw') else float('nan'), F.softplus(self.noise).item()

    #  def train_loop(self, epoch, train_loader, optimizer):
    #     avg_loss = 0
    #     pbar = tqdm.tqdm(train_loader)

    #     i = 0
    #     for x, _ in pbar:
    #         torch.save()
    #         i += 1
    #         self.n_query = x.size(1) - self.n_support
    #         if self.change_way:
    #             self.n_way = x.size(0)
    #         optimizer.zero_grad()
    #         loss = self.set_forward_loss(x)
    #         loss.backward()
    #         optimizer.step()
    #         avg_loss = avg_loss + loss.item()

    #         pbar.set_description(
    #             "Epoch {:d} | Loss {:f}".format(epoch, avg_loss / float(i + 1))
    #         )

    #     return avg_loss / i


    # def train_loop(self, epoch, train_loader, optimizer):
    #     avg_loss = 0
    #     pbar = tqdm.tqdm(train_loader)

    #     # print("***********",train_loader[0])############

    #     for iter in range(100):

    #         i = 0
    #         for x, _ in pbar:
    #             i += 1
    #             self.n_query = x.size(1) - self.n_support
    #             if self.change_way:
    #                 self.n_way = x.size(0)
    #             optimizer.zero_grad()
    #             loss = self.set_forward_loss(x)
    #             loss.backward()
    #             optimizer.step()
    #             avg_loss = avg_loss + loss.item()

    #             print("*****Loss is:*****",loss)

    #             if i == 1:
    #                 break

    #             # pbar.set_description(
    #             #     "Epoch {:d} | Loss {:f}".format(epoch, avg_loss / float(i + 1))
    #             # )
    #     return avg_loss / i

    # def train_loop(self, epoch, train_loader, optimizer ):
    #     print_freq = 10

    #     avg_loss=0
    #     for i, (x,_) in enumerate(train_loader):
    #         self.n_query = x.size(1) - self.n_support
    #         if self.change_way:
    #             self.n_way  = x.size(0)
    #         optimizer.zero_grad()
    #         loss = self.set_forward_loss( x )
    #         loss.backward()
    #         optimizer.step()
    #         avg_loss = avg_loss+loss.item()

    #         if i % print_freq==0:
    #             #print(optimizer.state_dict()['param_groups'][0]['lr'])
    #             print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))


    def test_loop(
        self,
        test_loader,
        record=None,
        return_std=False,
        use_progress=False,
        return_stats=False,
        feature_extractor=None
    ):
        acc_all = []
        logits_all = []
        targets_all = []
        results_all = []

        iter_num = len(test_loader)
        if use_progress:
            pbar = tqdm.tqdm(test_loader)
        else:
            pbar = test_loader
        mu_s, sigma_s, mu_pre, sigma_pre = [], [], [], []
        for x, _ in pbar:
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            # ---------------------------
            # TODO temporally replaced the call to correct() with the code
            # correct_this, count_this = self.correct(x)
            if feature_extractor is not None:
                x_flat = x.view(-1, *x.size()[2:])
                x_flat = feature_extractor(x_flat.cuda())
                x = x_flat.view(*x.size()[:2], -1)
            # print(x.shape) #torch.Size([5, 17, 3, 84, 84])
            with torch.no_grad():
                scores, mu_s_, sigma_s_, mu_pre_, sigma_pre_ = self.set_forward(x)
            mu_s.append(mu_s_); sigma_s.append(sigma_s_); mu_pre.append(mu_pre_); sigma_pre.append(sigma_pre_);
            logits_all.append(scores.cpu().detach().numpy())
            y_query = np.repeat(range(self.n_way), self.n_query)
            targets_all.append(y_query)
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == y_query)
            correct_this = float(top1_correct)
            count_this = len(y_query)
            # ---------------------------
            acc_all.append(correct_this / count_this * 100)

            if use_progress:
                pbar.set_description("Acc {:f}".format(np.mean(acc_all)))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        )
        print(torch.stack([torch.stack(mu_s, 0).mean(0),
                           torch.stack(sigma_s, 0).mean(0),
                           torch.stack(mu_pre, 0).mean(0),
                           torch.stack(sigma_pre, 0).mean(0),]).data.cpu().numpy())

        if return_std:
            return acc_mean, acc_std
        else:
            if return_stats:
                return {
                    "acc_mean": acc_mean,
                    "acc_std": acc_std,
                    "stats": {
                        "logits": torch.as_tensor(np.concatenate(logits_all, 0)),
                        "targets": torch.as_tensor(np.concatenate(targets_all, 0)),
                    },
                    # "results": results_all,
                }
            else:
                return acc_mean

    def set_forward_adaptation(
        self, x, is_feature=True
    ):  # further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, "Feature is fixed in further adaptation"
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(
            linear_clf.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001,
        )

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(
                    rand_id[i : min(i + batch_size, support_size)]
                ).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores

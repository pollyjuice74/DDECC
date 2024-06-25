"""
Implementation of "Denoising Diffusion Error Correction Codes" (DDECC), in ICLR23
https://arxiv.org/abs/2209.13533
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from DDECC import DDECCT



import tensorflow as tf
import torch
import numpy as np

import sionna as sn
from sionna.utils import BitErrorRate, BinarySource, ebnodb2no
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder



def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



class FEC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    
    def __len__(self):
        return self.len

    
    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        std_noise = random.choice(self.sigma)
        z = torch.randn(self.code.n) * std_noise
        #h = torch.from_numpy(np.random.rayleigh(1,self.code.n)).float()
        h=1
        y = h*bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()#, torch.tensor([std_noise]).float()


def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_samples = 0
    t = time.time()
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(
            train_loader):
        loss = model.loss(bin_to_sign(x))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        model.ema.update(model)
        ###
        cum_loss += loss.item() * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples


def test(model, device, test_loader_list, EbNo_range_test, min_FER=100, max_cum_count=1e7, min_cum_count=1e5):
    model.eval()
    test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_fer = cum_count = 0.
            _, x_pred_list, _, _ = model.p_sample_loop(next(iter(test_loader))[3])
            test_ber_ddpm , test_fer_ddpm = [0]*len(x_pred_list), [0]*len(x_pred_list)
            idx_conv_all = []
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                x_pred, x_pred_list, idx_conv,synd_all = model.p_sample_loop(y)
                x_pred = sign_to_bin(torch.sign(x_pred))

                idx_conv_all.append(idx_conv)
                for kk, x_pred_tmp in enumerate(x_pred_list):
                    x_pred_tmp = sign_to_bin(torch.sign(x_pred_tmp))

                    test_ber_ddpm[kk] += BER(x_pred_tmp, x) * x.shape[0]
                    test_fer_ddpm[kk] += FER(x_pred_tmp, x) * x.shape[0]
                    
                test_ber += BER(x_pred, x) * x.shape[0]
                test_fer += FER(x_pred, x) * x.shape[0]
                cum_count += x.shape[0]
                if (min_FER > 0 and test_fer > min_FER and cum_count > min_cum_count) or cum_count >= max_cum_count:
                    if cum_count >= 1e9:
                        logging.info(f'Cum count reached EbN0:{EbNo_range_test[ii]}')
                    else:    
                        logging.info(f'FER count treshold reached EbN0:{EbNo_range_test[ii]}')
                    break
            idx_conv_all = torch.stack(idx_conv_all).float()
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            for kk in range(len(test_ber_ddpm)):
                test_ber_ddpm[kk] /= cum_count
                test_fer_ddpm[kk] /= cum_count
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER_DDPM={test_ber_ddpm}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, -ln(BER)_DDPM={[-np.log(elem) for elem in test_ber_ddpm]}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, FER_DDPM={test_fer_ddpm}')
            logging.info(f'#It. to zero syndrome: Mean={idx_conv_all.mean()}, Std={idx_conv_all.std()}, Min={idx_conv_all.min()}, Max={idx_conv_all.max()}')
        ###
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, -np.log(elem)) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ber_list, test_loss_fer_list


def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    model = DDECCT(args, device=device,dropout=0).to(device)
    model.ema.register(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    EbNo_range_test = range(4, 7)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    train_dataloader = DataLoader(FEC_Dataset(code, std_train, len=args.batch_size * 1000, zero_cw=True), batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(FEC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(std_test))]
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss= train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
            logging.info(f'Model Saved')
        if epoch % (args.epochs//2) == 0 or epoch in [1,25]:
            test(model, device, test_dataloader_list, EbNo_range_test,min_FER=50,max_cum_count=1e6,min_cum_count=1e4)
    ############
    ############
    logging.info('Loading Best Model')
    model = torch.load(os.path.join(args.path, 'best_model')).to(device)
    logging.info('Regular Reverse Diffusion')
    test(model, device, test_dataloader_list, EbNo_range_test,min_FER=100)
    logging.info('Line Search Reverse Diffusion')
    model.line_search = True
    test(model, device, test_dataloader_list, EbNo_range_test,min_FER=100)


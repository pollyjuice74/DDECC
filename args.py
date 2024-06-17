import argparse
from  ECCT.Main import set_seed

import os
import torch
import logging
from datetime import datetime


def pass_args_ddecc():
    parser = argparse.ArgumentParser(description='PyTorch DDPM_ECCT')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='POLAR',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=32)
    parser.add_argument('--code_n', type=int, default=64)

    # model args
    parser.add_argument('--N_dec', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--h', type=int, default=8)

    # DDECC args
    parser.add_argument('--sigma', type=float, default=0.01)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    class Code():
        pass
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    G, H = Get_Generator_and_Parity(code)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    ###
    args.N_steps = code.pc_matrix.shape[0]+5 #number of diffusion steps
    ####################################################################
    model_dir = os.path.join('DDECCT_Results',
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) + '__' + datetime.now().strftime(
                                 "%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=print, format='%(message)s',
                        handlers=handlers)
    print(f"Path to model/logs: {model_dir}")
    print(args)

    return args


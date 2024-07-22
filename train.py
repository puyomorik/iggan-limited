""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import functools
from tqdm import tqdm

import torch

from gan_tools.model import utils
from gan_tools.train_utils.init_utils import model_initialize, log_initialize
import warnings

warnings.filterwarnings("ignore")

torch.cuda.current_device()
import torch.nn as nn

# Import my stuff
import gan_tools.train_utils.train_fns_aug as train_fns
from gan_tools.benchmark_utils import fid_score
from gan_tools.model.sync_batchnorm import patch_replication_callback
import sys
from gan_tools.model import BigGAN_new
from gan_tools.benchmark_utils import eval_utils


# The main training file. Config is a dictionary specifying the configuration
# of this training run.


def run(config):
    # Initialize the model for training
    device = 'cuda'
    model = BigGAN_new
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}
    G, D, G_ema, ema, GD1, GD2 = model_initialize(config, model, device, experiment_name,
                                                  state_dict)
    # Initialize the log file and the object
    test_log, train_log = log_initialize(config, experiment_name, state_dict)

    # Initialize the discriminator's data loader
    loaders = utils.get_data_loaders(**{**config, 'batch_size': (config['batch_size'] * config['num_D_steps']
                                                                 * config['num_D_accumulations']),
                                        'start_itr': state_dict['itr']})

    # y represents a categorical label for generator Typically, the generator does not require a data loader. Its
    # primary function is to receive random noise (and possibly categorical labels, if it is a conditional GAN) and
    # generate new data samples.
    z_, y_ = eval_utils.get_z_y(model, config)
    fixed_z, fixed_y = eval_utils.get_z_y(model, config)
    fixed_z.sample_()
    fixed_y.sample_()

    if not config['conditional']:
        fixed_y.zero_()
        y_.zero_()
    # Loaders are loaded, prepare the training function
    train = train_fns.GAN_training_function(G, D, GD1, GD2, z_, y_,
                                            ema, state_dict, config)
    # Prepare Sample function for use with inception metrics

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    print('Total training epochs ', config['num_epochs'])
    print("the dataset is ", config['dataset'], )
    if config['dataset'] == 'C10U' or config['dataset'] == 'C10':
        data_moments = './cifar_real_images/cifar10_10k_stats.npz'

    else:
        print("cannot find the dataset")
        sys.exit()

    print("the data moments is ", data_moments)
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(
                loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])
        for i, (x, y) in enumerate(pbar):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            # G.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            metrics = train(x, y)
            train_log.log(itr=int(state_dict['itr']), **metrics)

            # Every sv_log_interval, log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']),
                              **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']]
                                + ['%s : %+4.3f' % (key, metrics[key])
                                   for key in metrics]), end=' ')

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                          state_dict, config, experiment_name)

            # Test every specified interval
            # First load celeba moments

            experiment_name = (config['experiment_name'] if config['experiment_name']
                               else utils.name_from_config(config))
            if (not (state_dict['itr'] % config['test_every'])) and (epoch >= config['start_eval']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                # sampling images and saving to samples/experiments/epoch
                eval_model = G_ema if config['ema'] and config['use_ema'] else G
                fid = eval_utils.get_fid_score(eval_model, config, is_training=True, epoch=epoch)

                print("FID calculated")
                train_fns.update_FID(G, D, G_ema, state_dict, config, fid, experiment_name, test_log, epoch)
                # train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                #                get_inception_metrics, experiment_name, test_log)
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()

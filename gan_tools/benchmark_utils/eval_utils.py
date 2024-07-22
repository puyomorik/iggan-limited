import os
from typing import Any

import torch

from tqdm import trange
import numpy as np
import functools
from . import fid_score
from gan_tools.model import pickle_utils, utils, BigGAN_new


def load_model(config: dict[str, Any]):
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True

    # model = __import__(config['model'])
    G = BigGAN_new.Generator(**config).cuda()

    G.load_state_dict(torch.load(pickle_utils.open_file_or_url(config['network'])))
    if config['G_eval_mode']:
        G.eval()
    else:
        G.train()

    return G


def get_z_y(model, config: dict[str, Any]):
    batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(batch_size, model.dim_z, config['n_classes'],
                               device='cuda', fp16=config['G_fp16'],
                               z_var=config['z_var'])
    return z_, y_


def get_sample_func(model, config: dict[str, Any]):
    z_, y_ = get_z_y(model, config)
    sample_ = functools.partial(utils.sample, G=model, z_=z_, y_=y_, config=config)
    return sample_


def get_fid_score(model, config: dict[str, Any], is_training=False, epoch=0):
    batch_size = max(config['G_batch_size'], config['batch_size'])
    sample_ = get_sample_func(model, config)
    x, y = [], []
    print('Sampling %d images and saving them to npz...')

    for _ in trange(int(np.ceil(10000 / float(batch_size)))):
        with torch.no_grad():
            images, labels = sample_()
        x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
        y += [labels.cpu().numpy()]

    x = np.concatenate(x, 0)[:10000]
    y = np.concatenate(y, 0)[:10000]
    print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    if is_training:
        if not os.path.isdir(f'cifar_generated_images/{epoch}'):
            os.mkdir(f'cifar_generated_images/{epoch}')
        npz_filename = f'cifar_generated_images/{epoch}/samples.npz'
    else:
        npz_filename = 'cifar_generated_images/samples.npz'
    print('Saving npz to %s...' % npz_filename)

    np.savez(npz_filename, x=x, y=y)
    fid = fid_score.calculate_fid_given_paths(['./cifar_real_images/cifar10_10k_stats.npz',
                                               npz_filename],
                                              batch_size=50, cuda=True, dims=2048)
    return fid

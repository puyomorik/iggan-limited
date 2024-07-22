import torch
from torch import nn

from gan_tools.model import utils, BigGAN_new
from gan_tools.model.sync_batchnorm import patch_replication_callback


def model_initialize(config, model, device, experiment_name, state_dict):
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(
            config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GD1 = model.G_D1(G, D, config['conditional'])  # setting conditional to false
    GD2 = model.G_D2(G, D, config['conditional'])  # setting conditional to false

    print(G)
    print(D)
    print('Number of params in G: {} D: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    # Prepare state dict, which holds things like epoch # and itr #

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)

    # If parallel, parallelize the GD module
    if config['parallel']:
        GD1 = nn.DataParallel(GD1)
        GD2 = nn.DataParallel(GD2)
        if config['cross_replica']:
            patch_replication_callback(GD1)
            patch_replication_callback(GD2)

    return G, D, G_ema, ema, GD1, GD2


def log_initialize(config, experiment_name, state_dict):
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'],
                         experiment_name, config, state_dict)

    return test_log, train_log

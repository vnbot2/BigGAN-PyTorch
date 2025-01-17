""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).
    Let's go.
"""
import datetime
import time
import torch
import dataset
import BigGAN
import train_fns
import utils
from common import *
# IMG_SIZE = 64
# IMG_SIZE_2 = IMG_SIZE * 2

def run(config):
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = IMG_SIZE
    config['n_classes'] = 1
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'
    # Seed RNG
    utils.seed_rng(config['seed'])
    # Prepare root folders if necessary
    utils.prepare_root(config)
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else 'generative_dog_images')
    print('Experiment name is %s' % experiment_name)

    G = BigGAN.Generator(**config).to(device)
    
    D = BigGAN.Discriminator(**config).to(device)

    # if config['parallel']:
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(
            config['ema_decay']))
        G_ema = BigGAN.Generator(**{**config, 'skip_init': True,
                                    'no_optim': True}).to(device)
        G_ema = nn.DataParallel(G_ema)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    GD = BigGAN.G_D(G, D)
    print(G)
    print(D)
    print('Number of params in G: {} D: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)

    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] *
                    config['num_D_steps'] * config['num_D_accumulations'])
    loaders = dataset.get_data_loaders(
        data_root=config['data_root'],
        label_root=config['label_root'],
        batch_size=D_batch_size,
        num_workers=config['num_workers'],
        shuffle=config['shuffle'],
        pin_memory=config['pin_memory'],
        drop_last=True,
        load_in_mem=config['load_in_mem'],
        mask_out=config['mask_out']
    )

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    num_samples = config['num_fixed_samples']
    z_, y_ = utils.prepare_z_y(
        num_samples, G.module.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(
        num_samples, G.module.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    # Loaders are loaded, prepare the training function
    train = train_fns.create_train_fn(
        G, D, GD, z_, y_, ema, state_dict, config)

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    start_time = time.perf_counter()
    loader = loaders[0]
    total_iters = config['num_epochs'] * len(loader)
    # Train for specified number of epochs, although we mostly track G iterations.
    pbar = tqdm(total=total_iters)
    for _ in range(state_dict['itr']):
        pbar.update()
    timer = mmcv.Timer()
    timer.start()
    start_itr = state_dict['itr']
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        for i, data in enumerate(loader):
            x, y = data['img'], data['label']
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            x, y = x.to(device), y.to(device)
            metrics = train(x, y)

            if not (state_dict['itr'] % config['log_interval']):
                curr_time = timer.since_start()
                curr_time_str = datetime.datetime.fromtimestamp(
                    curr_time).strftime('%H:%M:%S')
                #       quang duong                     / (quang duong da di / thoi gian da di)
                eta = (
                    total_iters - state_dict['itr']) // ((state_dict['itr']-start_itr) / (curr_time+1))
                eta_str = datetime.datetime.fromtimestamp(
                    eta).strftime('%H:%M:%S')
                log = "[{}] [{}] [{} / {}] Ep {}, ".format(
                    curr_time_str, eta_str, state_dict['itr'], total_iters, epoch)
                log += ', '.join(['%s : %+4.3f' % (key, metrics[key])
                                  for key in metrics])

                pbar.set_description(log)
                # print(log)

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['sample_every']):
                if config['G_eval_mode']:
                    # print('Switching G to eval mode...')
                    G.eval()

                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                          state_dict, config, experiment_name, save_weight=False)

            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    # print('Switching G to eval mode...')
                    G.eval()

                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                          state_dict, config, experiment_name, save_weight=True)

            pbar.update()
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

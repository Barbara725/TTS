from torch.utils.data import DataLoader
from TTS.utils.audio import AudioProcessor

from TTS.utils.io import copy_model_files, load_config
\
from TTS.utils.training import setup_torch_training_env
from TTS.vocoder.datasets.gan_dataset import GANDataset
from TTS.vocoder.datasets.preprocess import load_wav_data, load_wav_feat_data

from torch.utils.data.distributed import DistributedSampler


use_cuda, num_gpus = setup_torch_training_env(False, False)


def setup_loader(ap, is_val=False, verbose=False):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = GANDataset(ap=ap,
                             items=eval_data if is_val else train_data,
                             seq_len=c.seq_len,
                             hop_len=ap.hop_length,
                             pad_short=c.pad_short,
                             conv_pad=c.conv_pad,
                             is_training=not is_val,
                             return_segments=not is_val,
                             use_noise_augment=c.use_noise_augment,
                             use_cache=c.use_cache,
                             verbose=verbose)
        dataset.shuffle_mapping()
        sampler = DistributedSampler(dataset, shuffle=True) if num_gpus > 1 else None
        loader = DataLoader(dataset,
                            batch_size=1 if is_val else c.batch_size,
                            shuffle=False if num_gpus > 1 else True,
                            drop_last=False,
                            sampler=sampler,
                            num_workers=c.num_val_loader_workers
                            if is_val else c.num_loader_workers,
                            pin_memory=False)
    return loader


def format_data(data):
    if isinstance(data[0], list):
        # setup input data
        c_G, x_G = data[0]
        c_D, x_D = data[1]

        # dispatch data to GPU
        if use_cuda:
            c_G = c_G.cuda(non_blocking=True)
            x_G = x_G.cuda(non_blocking=True)
            c_D = c_D.cuda(non_blocking=True)
            x_D = x_D.cuda(non_blocking=True)

        return c_G, x_G, c_D, x_D

    # return a whole audio segment
    co, x = data
    if use_cuda:
        co = co.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)
    return co, x, None, None



if __name__ == '__main__':

    VOCODER_CONFIG = "D:/Clone Project/TTS/TTS/vocoder/configs/multiband_melgan_config.json"

    # load configs
    c = load_config(VOCODER_CONFIG)

    #global train_data, eval_data

    eval_data, train_data = load_wav_data(c.data_path, c.eval_split_size)

    # setup audio processor
    ap = AudioProcessor(**c.audio)

    data_loader = setup_loader(ap, is_val=False, verbose=False)
    # print(data_loader)

    for num_iter, data in enumerate(data_loader):
        # format data
        c_G, y_G, c_D, y_D = format_data(data)
        # print('c_G',c_G[0])
        # print('----------------------------')
        # print('y_G',y_G[0])
        # print('---------------')
        # print('c_D',c_D[0])
        # print('-------------------------')
        # print('y_D',y_D[0])
        print(c_G.shape,y_G.shape)
        print(c_D.shape,y_D.shape)

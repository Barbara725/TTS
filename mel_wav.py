import numpy as np

from TTS.vocoder.utils.generic_utils import setup_generator
import os
import torch
from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.utils.audio import AudioProcessor

# # model paths
VOCODER_MODEL = "D:/TTS-master/TTS/Model/best_model.pth.tar"
VOCODER_CONFIG = "D:/TTS-master/TTS/vocoder/configs/multiband_melgan_config.json"

# # load configs
VOCODER_CONFIG = load_config(VOCODER_CONFIG)
#
use_cuda = False

# load vocoder model
vocoder_model = setup_generator(VOCODER_CONFIG)
vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
vocoder_model.remove_weight_norm()
vocoder_model.inference_padding = 0
# if use_cuda:
#     vocoder_model.cuda()
vocoder_model.eval()

# load the audio processor
# ap = AudioProcessor(**VOCODER_CONFIG['audio'])
ap = AudioProcessor(**VOCODER_CONFIG.audio)

# load wav data
folder = 'D:/vcc2020_database_training_source/source/SEF1/22050'
outpath = 'D:/vcc2020_database_training_source/source/SEF1_new'


floder_list = os.listdir(folder)#显示该文件夹下文件的名称
for lis in floder_list:
    pa = os.path.join(folder,lis)
    y = ap.load_wav(pa)
    mel = ap.melspectrogram(y)
    print(mel.shape)
    #mel_path = os.path.join(outpath, "mel")
    #np.save(mel_path, mel)

    use_griffin_lim = True
    if use_griffin_lim:
        wav = ap.inv_melspectrogram(mel)

        # if do_trim_silence:
        #     wav = ap.trim_silence

    path = os.path.join(outpath,"GLwav_{}".format(lis))
    ap.save_wav(wav, path)

    waveform = vocoder_model.inference(torch.FloatTensor(mel).unsqueeze(0))
    waveform = waveform.flatten()
    if use_cuda:
        waveform = waveform.cpu()
    waveform = waveform.numpy()
    # waveform = waveform.squeeze()
    path = os.path.join(outpath,"wav_{}".format(lis))
    # save the results
    ap.save_wav(waveform, path)


## melspectrogram和mel_postnet_spec





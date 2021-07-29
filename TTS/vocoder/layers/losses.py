import torch

from torch import nn
from torch.nn import functional as F


class TorchSTFT(nn.Module):#计算短时傅里叶频谱
    def __init__(self, n_fft, hop_length, win_length, window='hann_window'):
        """ Torch based STFT operation """
        super(TorchSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = nn.Parameter(getattr(torch, window)(win_length),
                                   requires_grad=False)# getattr() 函数用于返回一个对象属性值。

    def __call__(self, x):#函数调用。一个类实例要变成一个可调用对象，只需要实现一个特殊方法__call__()
        # B x D x T x 2 计算频谱,batchsize*频率*帧数*2（实部+虚部）
        o = torch.stft(x,
                       self.n_fft,
                       self.hop_length,
                       self.win_length,
                       self.window,
                       center=True,
                       pad_mode="reflect",  # compatible with audio.py
                       normalized=False,
                       onesided=True,
                       return_complex=False)
        M = o[:, :, :, 0]#实部
        P = o[:, :, :, 1]#虚部
        return torch.sqrt(torch.clamp(M ** 2 + P ** 2, min=1e-8))# 求幅值。clamp 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量。


#################################
# GENERATOR LOSSES
#################################


class STFTLoss(nn.Module):
    """ Single scale  STFT Loss """
    def __init__(self, n_fft, hop_length, win_length):
        super(STFTLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(n_fft, hop_length, win_length)#调用类

    def forward(self, y_hat, y):
        y_hat_M = self.stft(y_hat)#预测值stft
        y_M = self.stft(y)#目标值stft
        # log STFT magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))# 1norm l1_loss取各元素的绝对值差的平均值
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")#p='fro'时,Frobenius范数
        return loss_mag, loss_sc

class MultiScaleSTFTLoss(torch.nn.Module):
    """ Multi scale STFT loss """
    def __init__(self,
                 n_ffts=(1024, 2048, 512),
                 hop_lengths=(120, 240, 50),
                 win_lengths=(600, 1200, 240)):
        super(MultiScaleSTFTLoss, self).__init__()
        self.loss_funcs = torch.nn.ModuleList() #nn.ModuleList()储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。list

        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths): #zip并行遍历
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))#loss_funcs得到三个已实例化的类STFTLoss

    def forward(self, y_hat, y):#预测波形y_hat，目标波形y
        N = len(self.loss_funcs) #N=3
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:#循环每个类forward
            lm, lsc = f(y_hat, y)#输入类的参数
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N    #计算三组平均值
        loss_mag /= N
        return loss_mag, loss_sc #返回每组的mag对数幅度谱loss，sc谱收敛loss的平均值


class MultiScaleSubbandSTFTLoss(MultiScaleSTFTLoss):#计算每个子带的loss_mag, loss_sc
    """ Multiscale STFT loss for multi band model outputs """
    # pylint: disable=no-self-use
    def forward(self, y_hat, y): #shape[1]分解子带个数；shape[2]子带长度；shape[0]可能batchsize
        y_hat = y_hat.view(-1, 1, y_hat.shape[2])
        y = y.view(-1, 1, y.shape[2])#view 按要求改变y的形状,-1自动计算，取每条子带
        return super().forward(y_hat.squeeze(1), y.squeeze(1))#squeeze(1)代表若第二维度值为1则去除第二维度, super().forward调用父类前向方式;如（5，3，60）——>(15,1,60）——>(15,60)


class MSEGLoss(nn.Module):
    """ Mean Squared Generator Loss """
    # pylint: disable=no-self-use 生成器损失 '假' 数据的logits（即d_logits_fake），但所有的labels全设为1（即希望生成器generator输出1）。这样通过训练，生成器试图 ‘骗过’ discriminator。
    def forward(self, score_real):
        loss_fake = F.mse_loss(score_real, score_real.new_ones(score_real.shape))#希望D输出1；target=1
        return loss_fake


class HingeGLoss(nn.Module):
    """ Hinge Discriminator Loss """
    # pylint: disable=no-self-use
    def forward(self, score_real):
        # TODO: this might be wrong
        loss_fake = torch.mean(F.relu(1. - score_real))
        return loss_fake


##################################
# DISCRIMINATOR LOSSES
##################################


class MSEDLoss(nn.Module):
    """ Mean Squared Discriminator Loss """
    def __init__(self,):
        super(MSEDLoss, self).__init__()
        self.loss_func = nn.MSELoss()#nn.MSELoss()里面还是调用F.mse_loss

    # pylint: disable=no-self-use 判别器损失d_loss = d_loss_real + d_loss_fake
    def forward(self, score_fake, score_real):
        loss_real = self.loss_func(score_real, score_real.new_ones(score_real.shape)) #判为real，target=1输入真频谱
        loss_fake = self.loss_func(score_fake, score_fake.new_zeros(score_fake.shape))#假数据，target=0
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class HingeDLoss(nn.Module):
    """ Hinge Discriminator Loss """
    # pylint: disable=no-self-use
    def forward(self, score_fake, score_real):
        loss_real = torch.mean(F.relu(1. - score_real))
        loss_fake = torch.mean(F.relu(1. + score_fake))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class MelganFeatureLoss(nn.Module):#特征匹配损失
    def __init__(self,):
        super(MelganFeatureLoss, self).__init__()
        self.loss_func = nn.L1Loss()

    # pylint: disable=no-self-use
    def forward(self, fake_feats, real_feats):
        loss_feats = 0
        for fake_feat, real_feat in zip(fake_feats, real_feats):
            loss_feats += self.loss_func(fake_feat, real_feat)
        loss_feats /= len(fake_feats) + len(real_feats)
        return loss_feats


#####################################
# LOSS WRAPPERS
#####################################


def _apply_G_adv_loss(scores_fake, loss_func): #G的对抗损失 G_adv_loss (loss_func:self.mse_loss/hinge_loss)
    """ Compute G adversarial loss function
    and normalize values """
    adv_loss = 0
    if isinstance(scores_fake, list):#判断类型是否为list
        for score_fake in scores_fake:
            fake_loss = loss_func(score_fake)  #?
            adv_loss += fake_loss
        adv_loss /= len(scores_fake)
    else:
        fake_loss = loss_func(scores_fake)
        adv_loss = fake_loss
    return adv_loss


def _apply_D_loss(scores_fake, scores_real, loss_func):# D_loss,real,fake
    """ Compute D loss func and normalize loss values """
    loss = 0
    real_loss = 0
    fake_loss = 0
    if isinstance(scores_fake, list):
        # multi-scale loss
        for score_fake, score_real in zip(scores_fake, scores_real):
            total_loss, real_loss, fake_loss = loss_func(score_fake=score_fake, score_real=score_real)
            loss += total_loss
            real_loss += real_loss
            fake_loss += fake_loss
        # normalize loss values with number of scales
        loss /= len(scores_fake)
        real_loss /= len(scores_real)
        fake_loss /= len(scores_fake)
    else:
        # single scale loss
        total_loss, real_loss, fake_loss = loss_func(scores_fake, scores_real)
        loss = total_loss
    return loss, real_loss, fake_loss


##################################
# MODEL LOSSES
##################################
#根据训练参数配置，计算loss

class GeneratorLoss(nn.Module):
    def __init__(self, C):
        """ Compute Generator Loss values depending on training
        configuration """
        super(GeneratorLoss, self).__init__()
        assert not(C.use_mse_gan_loss and C.use_hinge_gan_loss)
        " [!] Cannot use HingeGANLoss and MSEGANLoss together."

        self.use_stft_loss = C.use_stft_loss
        self.use_subband_stft_loss = C.use_subband_stft_loss
        self.use_mse_gan_loss = C.use_mse_gan_loss
        self.use_hinge_gan_loss = C.use_hinge_gan_loss
        self.use_feat_match_loss = C.use_feat_match_loss

        self.stft_loss_weight = C.stft_loss_weight
        self.subband_stft_loss_weight = C.subband_stft_loss_weight
        self.mse_gan_loss_weight = C.mse_G_loss_weight
        self.hinge_gan_loss_weight = C.hinge_G_loss_weight
        self.feat_match_loss_weight = C.feat_match_loss_weight

        if C.use_stft_loss:
            self.stft_loss = MultiScaleSTFTLoss(**C.stft_loss_params)
        if C.use_subband_stft_loss:
            self.subband_stft_loss = MultiScaleSubbandSTFTLoss(**C.subband_stft_loss_params)
        if C.use_mse_gan_loss:
            self.mse_loss = MSEGLoss()
        if C.use_hinge_gan_loss:
            self.hinge_loss = HingeGLoss()
        if C.use_feat_match_loss:
            self.feat_match_loss = MelganFeatureLoss()

    def forward(self, y_hat=None, y=None, scores_fake=None, feats_fake=None, feats_real=None, y_hat_sub=None, y_sub=None):#设为None的好处，若有值则赋予，无值为none
        gen_loss = 0
        adv_loss = 0
        return_dict = {}

        # 多带=子带+全带
        # STFT Loss
        if self.use_stft_loss:
            stft_loss_mg, stft_loss_sc = self.stft_loss(y_hat.squeeze(1), y.squeeze(1))
            return_dict['G_stft_loss_mg'] = stft_loss_mg
            return_dict['G_stft_loss_sc'] = stft_loss_sc
            gen_loss += self.stft_loss_weight * (stft_loss_mg + stft_loss_sc)#stft_loss_weight=0.5

        # subband STFT Loss
        if self.use_subband_stft_loss:
            subband_stft_loss_mg, subband_stft_loss_sc = self.subband_stft_loss(y_hat_sub, y_sub)
            return_dict['G_subband_stft_loss_mg'] = subband_stft_loss_mg
            return_dict['G_subband_stft_loss_sc'] = subband_stft_loss_sc
            gen_loss += self.subband_stft_loss_weight * (subband_stft_loss_mg + subband_stft_loss_sc) #subband_stft_loss_weight": 0.5

        # multiscale MSE adversarial loss
        if self.use_mse_gan_loss and scores_fake is not None:
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.mse_loss)
            return_dict['G_mse_fake_loss'] = mse_fake_loss
            adv_loss += self.mse_gan_loss_weight * mse_fake_loss #mse_G_loss_weight": 2.5,

        # multiscale Hinge adversarial loss
        if self.use_hinge_gan_loss and not scores_fake is not None:
            hinge_fake_loss = _apply_G_adv_loss(scores_fake, self.hinge_loss)
            return_dict['G_hinge_fake_loss'] = hinge_fake_loss
            adv_loss += self.hinge_gan_loss_weight * hinge_fake_loss

        # Feature Matching Loss
        if self.use_feat_match_loss and not feats_fake:
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            return_dict['G_feat_match_loss'] = feat_match_loss
            adv_loss += self.feat_match_loss_weight * feat_match_loss
        return_dict['G_loss'] = gen_loss + adv_loss
        return_dict['G_gen_loss'] = gen_loss
        return_dict['G_adv_loss'] = adv_loss
        return return_dict


class DiscriminatorLoss(nn.Module):
    """ Compute Discriminator Loss values depending on training
    configuration """
    def __init__(self, C):
        super(DiscriminatorLoss, self).__init__()
        assert not(C.use_mse_gan_loss and C.use_hinge_gan_loss),\
            " [!] Cannot use HingeGANLoss and MSEGANLoss together."

        self.use_mse_gan_loss = C.use_mse_gan_loss
        self.use_hinge_gan_loss = C.use_hinge_gan_loss

        if C.use_mse_gan_loss:
            self.mse_loss = MSEDLoss()
        if C.use_hinge_gan_loss:
            self.hinge_loss = HingeDLoss()

    def forward(self, scores_fake, scores_real):
        loss = 0
        return_dict = {}

        if self.use_mse_gan_loss:
            mse_D_loss, mse_D_real_loss, mse_D_fake_loss = _apply_D_loss(
                scores_fake=scores_fake,
                scores_real=scores_real,
                loss_func=self.mse_loss)
            return_dict['D_mse_gan_loss'] = mse_D_loss
            return_dict['D_mse_gan_real_loss'] = mse_D_real_loss
            return_dict['D_mse_gan_fake_loss'] = mse_D_fake_loss
            loss += mse_D_loss

        if self.use_hinge_gan_loss:
            hinge_D_loss, hinge_D_real_loss, hinge_D_fake_loss = _apply_D_loss(
                scores_fake=scores_fake,
                scores_real=scores_real,
                loss_func=self.hinge_loss)
            return_dict['D_hinge_gan_loss'] = hinge_D_loss
            return_dict['D_hinge_gan_real_loss'] = hinge_D_real_loss
            return_dict['D_hinge_gan_fake_loss'] = hinge_D_fake_loss
            loss += hinge_D_loss

        return_dict['D_loss'] = loss
        return return_dict

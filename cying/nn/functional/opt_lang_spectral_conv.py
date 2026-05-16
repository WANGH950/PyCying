import torch


def opt_lang_spectral_conv(input, weight):
    output = torch.fft.rfft(input)
    return torch.fft.irfft(weight * output)
import torch
import torch.utils
import torch.utils.data


def select(img, act):
    if act == 0:
        return brightness(img, 0.05, 0)
    elif act == 1:
        return brightness(img, 0.05, 1)
    elif act == 2:
        return brightness(img, 0.05, 2)
    elif act == 3:
        return brightness(img, -0.05, 0)
    elif act == 4:
        return brightness(img, 0.05)
    elif act == 5:
        return contrast(img, 0.894, 0)
    elif act == 6:
        return contrast(img, 0.894, 1)
    elif act == 7:
        return contrast(img, 0.894, 2)
    elif act == 8:
        return contrast(img, 1.414, 0)
    elif act == 9:
        return contrast(img, 1.414, 1)
    elif act == 10:
        return contrast(img, 1.414, 2)
    elif act == 11:
        return contrast(img, 0.894)
    elif act == 12:
        return contrast(img, 1.414)
    elif act == 13:
        return brightness(img, 0.005, 0)
    elif act == 14:
        return brightness(img, 0.005, 1)
    elif act == 15:
        return brightness(img, 0.005, 2)
    elif act == 16:
        return brightness(img, -0.005, 0)
    elif act == 17:
        return brightness(img, 0.005)
    elif act == 18:
        return gamma_corr(img, 0.6, 0)
    elif act == 19:
        return gamma_corr(img, 0.6, 1)
    elif act == 20:
        return gamma_corr(img, 0.6, 2)
    elif act == 21:
        return gamma_corr(img, 1.05, 0)
    elif act == 22:
        return gamma_corr(img, 1.05, 1)
    elif act == 23:
        return gamma_corr(img, 1.05, 2)
    elif act == 24:
        return gamma_corr(img, 0.6)
    elif act == 25:
        return gamma_corr(img, 1.05)


def gamma_corr(image, gamma, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = mod[:, channel, :, :] ** gamma
    else:
        mod = mod ** gamma
    return mod


def brightness(image, bright, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(mod[:, channel, :, :] + bright, 0, 1)
    else:
        mod = torch.clamp(mod + bright, 0, 1)
    return mod


def contrast(image, alpha, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(
            torch.mean(mod[:, channel, :, :]) + alpha * (mod[:, channel, :, :] - torch.mean(mod[:, channel, :, :])), 0,
            1)
    else:
        mod = torch.clamp(torch.mean(mod) + alpha * (mod - torch.mean(mod)), 0, 1)
    return mod

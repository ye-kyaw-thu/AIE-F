import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributedTorch(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributedTorch, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class Conv2dPadded(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dPadded, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.filter_height, self.filter_width = kernel_size
        self.stride_height, self.stride_width = stride

    def forward(self, x):
        _, _, in_height, in_width = x.shape
        pad_along_height = max(self.filter_height - self.stride_height, 0) if (in_height % self.stride_height == 0) else max(self.filter_height - (in_height % self.stride_height), 0)
        pad_along_width = max(self.filter_width - self.stride_width, 0) if (in_width % self.stride_width == 0) else max(self.filter_width - (in_width % self.stride_width), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class MOSNet(nn.Module):
    def __init__(self, frame_length=1024, dropout=0.0, device='cuda'):
        super(MOSNet, self).__init__()
        self.device = device
        self.frame_length = frame_length
        self.crop_samples = frame_length - 4
        self.hann_win = torch.hann_window(self.frame_length).to(self.device)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            Conv2dPadded(16, 16, (3, 3), (1, 3)), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            Conv2dPadded(32, 32, (3, 3), (1, 3)), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            Conv2dPadded(64, 64, (3, 3), (1, 3)), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)), nn.ReLU(),
            Conv2dPadded(128, 128, (3, 3), (1, 3)), nn.ReLU()
        )

        self.blstm1 = nn.LSTM(7 * 128, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.flatten = TimeDistributedTorch(nn.Flatten(), batch_first=True)
        self.dense1 = nn.Sequential(
            TimeDistributedTorch(nn.Sequential(nn.Linear(256, 128), nn.ReLU()), batch_first=True),
            nn.Dropout(dropout)
        )
        self.frame_layer = TimeDistributedTorch(nn.Linear(128, 1), batch_first=True)
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def _compute_stft(self, forward_input):
        stft = torch.stft(
            forward_input,
            n_fft=self.frame_length,
            hop_length=self.frame_length // 4,
            win_length=self.frame_length,
            window=self.hann_win,
            return_complex=True
        )
        stft = stft.abs().float()
        return stft[..., 2:-2].unsqueeze(1).permute((0, 1, 3, 2))

    def getFtrMaps(self, forward_input):
        x = self._compute_stft(forward_input)
        outputs = []
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            x = conv(x)
            outputs.append(x)
        return outputs

    def forward(self, forward_input):
        x = self._compute_stft(forward_input)
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            x = conv(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 7 * 128)
        x, _ = self.blstm1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        frame_score = self.frame_layer(x)
        avg_score = self.average_layer(frame_score.permute(0, 2, 1))
        return avg_score.view(avg_score.size(0), -1), frame_score


class MOSNetLoss(nn.Module):
    def __init__(self, model_weights, frame_length=1024, device='cuda'):
        super(MOSNetLoss, self).__init__()
        self.device = device
        self.mosnet = MOSNet(frame_length=frame_length, device=device)
        self.mosnet.load_state_dict(torch.load(model_weights, map_location=device))
        self.mosnet.to(device)
        self.mosnet.train()

    def ftrLoss(self, x, ref):
        ftrs_x = self.mosnet.getFtrMaps(x)
        ftrs_ref = self.mosnet.getFtrMaps(ref)
        loss = sum(F.l1_loss(fx, fr).mean() for fx, fr in zip(ftrs_x, ftrs_ref)) / len(ftrs_x)
        return loss

    def forward(self, x, ref=None):
        x, ref = x.to(self.device), ref.to(self.device) if ref is not None else (x.to(self.device), None)
        avg_mos_score, mos_score = self.mosnet(x)
        if ref is not None:
            avg_mos_score_ref, mos_score_ref = self.mosnet(ref)
            return ((mos_score_ref - mos_score) ** 2).mean()
        return 5.0 - avg_mos_score.mean()
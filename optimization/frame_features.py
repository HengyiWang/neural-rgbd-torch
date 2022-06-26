import torch
import torch.nn as nn

class FeatureArray(nn.Module):
    """
    Per-frame corrective latent code.
    """

    def __init__(self, num_frames, num_channels):
        #super(FeatureArray, self).__init__()

        self.num_frames = num_frames
        self.num_channels = num_channels

        self.data = nn.Parameter(
            torch.random.normal([num_frames, num_channels], dtype=torch.float32)
        )

    def forward(self, ids):
        # get the corr
        ids = torch.where(ids < self.num_frames, ids, torch.zeros_like(ids))
        return torch.gather(self.data, ids)
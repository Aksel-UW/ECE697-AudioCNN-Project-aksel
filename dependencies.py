print("This code was adapted from the code found at the github page https://github.com/musikalkemist/pytorchforaudio/tree/main/10%20Predictions%20with%20sound%20classifier and through the video tutorials found at https://www.youtube.com/watch?v=gp2wZqDoJ1Y&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm")

import torch

from torch import nn
from torchinfo import summary

import torchaudio

import os
from torch.utils.data import Dataset
import pandas as pd
import soundfile

from torch.utils.data import DataLoader
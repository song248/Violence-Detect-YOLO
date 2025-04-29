import os

import torch
import torch.nn as nn
from fight_module.util import *
import dotenv

dotenv.load_dotenv()

# LSTM 기반 모델 정의
class FightLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, num_layers=2, dropout=0.5):
        super(FightLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 시점의 hidden state 사용
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()

class FightDetector:
    """
    Fight detection module using an LSTM model.
    """

    def __init__(self, fight_model, fps=30, sequence_length=5):
        # Load pre-trained LSTM model
        self.input_size = 16
        self.model = FightLSTM()
        self.model.load_state_dict(torch.load(fight_model))
        self.model.eval()  # Set to evaluation mode

        # Define keypoints for calculating angles
        self.coordinate_for_angel = [
            [8, 6, 2],      # right shoulder - right elbow - right wrist
            [11, 5, 7],     # left shoulder - left elbow - left wrist
            [6, 8, 10],     # right elbow - right wrist - right hand
            [5, 7, 9],      # left elbow - left wrist - left hand
            [6, 12, 14],    # right elbow - right hip - right knee
            [5, 11, 13],    # left elbow - left hip - left knee
            [12, 14, 16],   # right hip - right knee - right ankle
            [11, 13, 15]    # left hip - left knee - left ankle
        ]

        # Frame sampling settings
        self.sequence_length = sequence_length
        self.sampling_interval = fps // self.sequence_length
        self.features_buffer = []
        self.frame_counter = 0

    def detect(self, conf, xyn):
        """
        Detects fight action based on keypoints and confidence scores using LSTM.

        Args:
            conf (list): Confidence scores for each keypoint.
            xyn (list): Coordinates (x, y, visibility) for each keypoint.

        Returns:
            bool: True if fight detected, False otherwise.
        """
        self.frame_counter += 1

        if self.frame_counter % self.sampling_interval != 0:
            return False

        input_list = []
        for n in self.coordinate_for_angel:
            first, mid, end = n[0], n[1], n[2]

            c1, c2, c3 = xyn[first], xyn[mid], xyn[end]

            if is_coordinate_zero(c1, c2, c3):
                return False

            input_list.append(calculate_angle(c1, c2, c3))
            conf1, conf2, conf3 = conf[first], conf[mid], conf[end]
            input_list.append(torch.mean(torch.Tensor([conf1, conf2, conf3])).item())

        self.features_buffer.append(input_list)

        if len(self.features_buffer) < self.sequence_length:
            return False

        if len(self.features_buffer) > self.sequence_length:
            self.features_buffer.pop(0)

        input_tensor = torch.Tensor([self.features_buffer])  # shape: (1, 5, 16)
        with torch.no_grad():
            prediction = self.model(input_tensor)
        return prediction.item() > 0.5
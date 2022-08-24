import dependencies

import torch
import torchaudio

from networks import CNNNetwork
from urbansound8kloader import UrbanSoundDataset
from networktraining import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

#This code was adapted from the code found at the github page https://github.com/musikalkemist/pytorchforaudio/tree/main/10%20Predictions%20with%20sound%20classifier and through the video tutorials found at https://www.youtube.com/watch?v=gp2wZqDoJ1Y&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")


    # get a sample from the urban sound dataset for inference
    input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
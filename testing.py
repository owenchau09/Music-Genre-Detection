import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import librosa
from PIL import Image


######################################################################### IMPORTANT
# Input is as follows:python (pathof testing.py) --path (pathofsong(wav))
# Example: python "C:\Users\Owen Chau\Downloads\testing.py" --path "C:\Users\Owen Chau\Downloads\neural_net\Eminem - Without Me (Audio).wav"
#########################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, 6)  #6 labels

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 3380)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

model = Net()
model.load_state_dict(torch.load('C:/Users/Owen Chau/Downloads/tensorv2.pth'))  #CHANGE TO PATH OF WHERE YOU DOWNLOADED tensorv2.pth (Remember to use forward slashes)
model.eval()

################convert .wav -> image
file, samplingRate = librosa.load(args.path)


# Specify the duration in seconds
duration = 10.0
start_time = 5.0

# Calculate the number of samples corresponding to the specified duration
num_samples = int(samplingRate * duration)
start_idx = int(samplingRate * start_time)

# Extract only the first 'num_samples' from the audio file
example = file[start_idx:start_idx+num_samples]
hopLength = 512 # the number of samples between successive columns of the spectrogram

spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y = example, sr = samplingRate, n_fft = 2048, hop_length = hopLength, n_mels = 128, power = 4.0), ref = np.max)
plt.figure()
librosa.display.specshow(spectrogram, sr = samplingRate, hop_length = hopLength, x_axis = "off", y_axis = "off")

# Save the spectrogram as an RGBA image in-memory (no file is saved)
buffer = plt.gcf()
buffer.canvas.draw()
img = np.array(buffer.canvas.renderer._renderer)
plt.close()

# Find the coordinates of non-white pixels
non_white_pixels = np.any(img != [255, 255, 255, 255], axis=-1)
rows, cols = np.where(non_white_pixels)
# Crop the image to the bounding box of non-white pixels
img = img[min(rows):max(rows) + 1, min(cols):max(cols) + 1, :]
################


################resize the image -> (4, 64, 64)
new_height = 64
new_width = 64

# Calculate the resize factors
resize_factor_height = new_height / img.shape[0]
resize_factor_width = new_width / img.shape[1]

# Resize the image using scipy.ndimage.zoom
resized_img = zoom(img, (resize_factor_height, resize_factor_width, 1))
plt.imsave('C:/Users/Owen Chau/Downloads/neural_net/audio_waveform3.png', resized_img) #CHANGE PATH TO WHERE YOU DOWNLOADED audio_waveform3.png (remember forward slashes)
transposed_img = np.transpose(resized_img.astype(np.float32), (2,0,1)) / 255.

output = model(torch.from_numpy(transposed_img))
output = nn.Softmax()(output)
output = output.detach().cpu().numpy()
print(output[0])
predicted_genre = np.argmax(output)
if predicted_genre == 0:
    print("Most likely: classical")
if predicted_genre == 1:
    print("Most likely: rock")
if predicted_genre == 2:
    print("Most likely: blues")
if predicted_genre == 3:
    print("Most likely: jazz")
if predicted_genre == 4:
    print("Most likely: pop")
if predicted_genre == 5:
    print("Most likely: hiphop")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # We use tensorflow for training and testing ANN
import pyedflib

from read_input_data import ANN_examples_3channels_markup # This is needed if we work with 2 channels of ECoG
from read_input_data import ANN_examples_2channels_markup # This is needed if we work with 3 channels of ECoG

############################
#       Parameters
############################
number_of_channels = 3          # number of channels
input_size=number_of_channels*2 # number of input neurons = number of channels * 2
considered_channels = [1,2,3]     # channel numbers to be imported from the .edf file. values are from 1 to 3. Can be empty if number_of_channels=3
dt_average_s=1                  # window shift in seconds
window_average_s=10             # averaging window in seconds

Tmax=-1                       # Maximal time in seconds. =-1 for automatic selection
Tmin=-1                       # Maximal time in seconds. =-1 for automatic selection

ANN_file = "trained_ANNs/model_averaged_123" # path and name of the file with trained ANN
edf_file = "examples/example1.edf" # path and name of the edf file
file_markup = "examples/example1_markup.dat" # File with wavelet based markup. If not exist, set = None
#file_markup = None


############################
#    Edf-file proceeding
############################
print("Edf-file proceeding...")
f = pyedflib.EdfReader(edf_file)
signal_labels = f.getSignalLabels()
print(signal_labels)
n = f.signals_in_file
T0=f.getStartdatetime()
freq = int(f.getSampleFrequency(0))
print("Frequency = ",freq," Hz")
f_len = np.size(f.readSignal(0),0) # number of points in each channel
print(str(f_len) + " points")
imported_data = np.zeros((f_len,4))

########################################
#    Generating an input ECoG array
########################################

imported_data[:,0] = np.arange(0,f_len/freq,1/freq) # 1st column = time in seconds
if number_of_channels == 2 :
    imported_data[:, 1] = np.array(f.readSignal(considered_channels[0]-1))
    imported_data[:, 2] = np.array(f.readSignal(considered_channels[1]-1))
else :
    imported_data[:,1] = np.array(f.readSignal(0)) # 1st channel (FrL)
    imported_data[:,2] = np.array(f.readSignal(1)) # 2nd channel (FrR)
    imported_data[:,3] = np.array(f.readSignal(2)) # 3rd channel (OcR)

########################################
#    Generating an input ECoG array
########################################
if number_of_channels == 2 :
    # Import files and calculate the mean and standard deviation for 2 channels:
    Time_average_test, x_test, y_test = ANN_examples_2channels_markup(imported_data, file_markup, Tmin, Tmax, dt_average_s,window_average_s, input_size)
else :
    # Import files and calculate the mean and standard deviation for all 3 channels:
    Time_average_test, x_test, y_test = ANN_examples_3channels_markup(imported_data,file_markup,Tmin,Tmax,dt_average_s,window_average_s,input_size)

######################
#    Testing ANN
######################
model = tf.keras.models.load_model(ANN_file)  # Import already trained ANN
ANN_predictions_test = model.predict(x_test)                    # Recognition from ANN with real numbers from 0 to 1.

# Making a prediction in binary form 1-sleeping, 0-awake.
ANN_predictions_test_bool = np.zeros(np.shape(ANN_predictions_test))
ANN_predictions_test_bool[ANN_predictions_test>=0.5] = 1
accuracy_test = model.evaluate(x_test, y_test)
print(edf_file)
print("Testing Loss:", accuracy_test[0])     # Loss on testing data, based on existing markup
print("Testing Accuracy:", accuracy_test[1]) # Accuracy on testing data, based on existing markup

# Making figure
plt.rcParams['figure.figsize'] = 10,4
plt.plot(Time_average_test[:,0], ANN_predictions_test_bool[:,0], color='lavender')
plt.plot(Time_average_test[:,0], y_test[:,0], color='navy')
plt.plot(Time_average_test[:,0], ANN_predictions_test[:,0], color='darkorange')
plt.legend(['ANN binary', 'Given markup', 'ANN resp.'])
plt.xlabel('time')
plt.ylabel('Sleep markup')
plt.title(edf_file)

# Saving figure
plt.savefig("temp.png")
#plt.show()
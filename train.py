from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.models import Sequential
import numpy as np
import random
import os
import tensorflow as tf
# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dropout

from common.numpy_fast import interp
from torque_model.helpers import LatControlPF, TORQUE_SCALE, random_chance, STATS_KEYS, REVERSED_STATS_KEYS, MODEL_INPUTS, normalize_sample, normalize_value
from torque_model.load import load_data
from sklearn.model_selection import train_test_split
from selfdrive.config import Conversions as CV
import seaborn as sns
from common.basedir import BASEDIR

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.chdir(os.path.join(BASEDIR, 'torque_model'))

# print(tf.config.optimizer.get_experimental_options())
# tf.config.optimizer.set_experimental_options({'constant_folding': True, 'pin_to_host_optimization': True, 'loop_optimization': True, 'scoped_allocator_optimization': True})
# print(tf.config.optimizer.get_experimental_options())

try:
  tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

to_normalize = False
data, data_sequences, data_stats, _ = load_data('data', to_normalize)
# del data_high_delay, data_sequences
print(f'Number of samples: {len(data)}')

x_train = []
for line in data:
  x_train.append([line[inp] for inp in MODEL_INPUTS])

y_train = []
for line in data:
  # the torque key is set by the data loader, it can come from torque_eps or torque_cmd depending on engaged status
  y_train.append(line['torque'])

print(f'Output (torque) min/max: {[min(y_train), max(y_train)]}')

# x_train = []  # only use synthetic samples
# y_train = []

# sns.distplot([abs(line['steering_angle']) for line in data], bins=200)
# plt.title('steering angle')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([abs(line['steering_rate']) for line in data], bins=200)
# plt.title('steering rate')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([line['v_ego'] for line in data], bins=200)
# plt.title('speed')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([abs(line['torque']) for line in data], bins=200)
# plt.title('torque')
# plt.pause(0.01)
# input()


x_train = np.array(x_train)
y_train = np.array(y_train) / TORQUE_SCALE

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.35)
print(x_train.shape)
print('Training on {} samples and validating on {} samples'.format(len(x_train), len(x_test)))


model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(8, activation=LeakyReLU()))
# model.add(Dropout(1/8))
model.add(Dense(16, activation=LeakyReLU()))
# model.add(Dropout(1/16))
# model.add(Dense(24, activation=LeakyReLU()))
model.add(Dense(1))

epochs = 150
starting_lr = .01
ending_lr = 0.001
decay = (starting_lr - ending_lr) / epochs

# opt = Adam(learning_rate=starting_lr, amsgrad=True, decay=decay)
opt = Adadelta(learning_rate=1)
# opt = Adagrad(learning_rate=0.2)
model.compile(opt, loss='mae', metrics='mse')
try:
  model.fit(x_train, y_train, batch_size=512, epochs=20, validation_data=(x_test, y_test))
  model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))
  model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_test, y_test))
  model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=32, epochs=25, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))
except KeyboardInterrupt:
  pass


def plot_random_samples():
  idxs = np.random.choice(range(len(x_train)), 50)
  x_test = x_train[idxs]
  y_test = y_train[idxs].reshape(-1) * TORQUE_SCALE
  pred = model.predict(np.array([x_test])).reshape(-1) * TORQUE_SCALE

  plt.figure(0)
  plt.clf()
  plt.plot(y_test, label='ground truth')
  plt.plot(pred, label='prediction')
  plt.legend()
  plt.show()


# plot_random_samples()

# speed in mph, accel in m/s/s. bit weird, but easy to work with
def plot_response(angle=15, around=15, speed=37, accel=0):  # plots model output compared to pid on steady angle but changing desired angle
  # the two lines should ideally be pretty close
  plt.figure(2)
  plt.clf()
  desired = np.linspace(angle - around, angle + around, 200)
  error = np.array(desired) - angle
  rate = normalize_value(0, 'rate', data_stats, to_normalize)
  speed *= CV.MPH_TO_MS
  y_pid = []
  y_model = []
  for des in desired:
    y_model.append(
      model.predict_on_batch(np.array([[normalize_value(des, "angle", data_stats, to_normalize),
                                        normalize_value(angle, "angle", data_stats, to_normalize),
                                        rate, rate,
                                        normalize_value(speed, "speed", data_stats, to_normalize),
                                        normalize_value(accel, "accel", data_stats, to_normalize)]
                                       ]))[0][0] * 1500)
    y_pid.append(pid.update(des, angle, speed) * 1500)
  plt.plot(error, y_pid, label='standard pf controller')
  plt.plot(error, y_model, label='model')
  plt.plot([0] * len(y_pid), np.linspace(max(y_model), min(y_model), len(y_pid)))
  plt.xlabel('angle error')
  plt.ylabel('torque')
  plt.legend()
  plt.show()


pid = LatControlPF()
def plot_sequence(sequence_idx=3, show_controller=True):  # plots what model would do in a sequence of data
  sequence = data_sequences[sequence_idx]

  plt.figure()
  plt.clf()

  ground_truth = [line['torque'] for line in sequence]
  plt.plot(ground_truth, label='ground truth')

  _x = [normalize_sample(line, data_stats, to_normalize) for line in sequence]
  _x = [[line[inp] for inp in MODEL_INPUTS] for line in _x]
  pred = model.predict(np.array(_x)).reshape(-1) * TORQUE_SCALE
  plt.plot(pred, label='prediction')

  if show_controller:
    controller = [pid.update(line['fut_steering_angle'], line['steering_angle'], line['v_ego']) * TORQUE_SCALE for line in sequence]  # what a pf controller would output
    plt.plot(controller, label='standard controller')

  plt.legend()
  plt.show()

plot_sequence(-3)
plot_sequence(4)
plot_sequence(8)

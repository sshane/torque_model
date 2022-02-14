import copy
import os
import random
from objprint import add_objprint

import matplotlib.pyplot as plt
import numpy as np

from common.numpy_fast import interp
from selfdrive.config import Conversions as CV
import seaborn as sns
import time
import pickle
from torque_model.helpers import feedforward, random_chance, TORQUE_SCALE, LatControlPF, STATS_KEYS, REVERSED_STATS_KEYS, MODEL_INPUTS, normalize_sample
from common.basedir import BASEDIR

DT_CTRL = 0.01
MIN_SAMPLES = 5 / DT_CTRL  # seconds to frames
# STATS_KEYS = {'steering_angle': 'angle', 'steering_rate': 'rate', 'v_ego': 'speed', 'torque': 'torque'}  # this renames keys to shorter names to access later quicker

os.chdir(os.path.join(BASEDIR, 'torque_model'))


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def get_steer_delay(speed):
  return round(interp(speed, [20 * CV.MPH_TO_MS, 72 * CV.MPH_TO_MS], [32, 52]))


def offset_torque(_data):  # todo: offsetting both speed and accel seem to decrease model loss by a LOT. maybe we should just offset all gas instead of these two todo: maybe not?
  for i in range(len(_data)):  # accounts for steer actuator delay (from torque to change in angle)
    steering_angle = [line['steering_angle'] for line in _data[i]]
    steering_rate = [line['steering_rate'] for line in _data[i]]
    data_len = len(_data[i])
    steer_delay = 0
    for j in range(data_len):
      steer_delay = get_steer_delay(_data[i][j]['v_ego'])  # interpolate steer delay from speed
      if j + steer_delay >= data_len:
        break
      _data[i][j]['fut_steering_angle'] = float(steering_angle[j + steer_delay])
      _data[i][j]['fut_steering_rate'] = float(steering_rate[j + steer_delay])
    _data[i] = _data[i][:-steer_delay]  # removes trailing samples (uses last steer delay)
  return _data


def filter_data(_data):
  # KEEP_DATA = 'engaged'  # user, engaged, or all
  keep_distribution = {'engaged': 50, 'user': 50}

  def sample_ok(_line):
    return 1 * CV.MPH_TO_MS < _line['v_ego'] and -4 < _line['a_ego'] < 3 and abs(_line['steering_rate']) < 300 and \
           abs(_line['fut_steering_rate']) < 300 and abs(_line['torque_eps']) < 3000

  filtered_sequences = []
  for sequence in _data:
    filtered_seq = []
    for line in sequence:
      if not sample_ok(line):
        continue

      if line['engaged'] and random_chance(keep_distribution['engaged']):  # and random_chance(15):
        line['torque'] = line['torque_cmd']
        filtered_seq.append(line)
      if not line['engaged'] and random_chance(keep_distribution['user']):
        line['torque'] = line['torque_eps'] + line['torque_driver']  # fixme: do we add these? (old: i think eps makes more sense than driver)
        filtered_seq.append(line)

    if len(filtered_seq):
      filtered_sequences.append(filtered_seq)

  return filtered_sequences

  # flattened = [i for j in filtered_sequences for i in j]
  # return [i for j in filtered_sequences for i in j], filtered_sequences


def even_out_torque(_data):
  threshold = 100
  left_curve_samples = len([line for line in _data if line['torque'] > threshold])
  right_curve_samples = len([line for line in _data if line['torque'] < -threshold])
  print('Left curve samples: {}'.format(left_curve_samples))
  print('Right curve samples: {}'.format(right_curve_samples))
  left_curve_weight = right_curve_samples / left_curve_samples if left_curve_samples > right_curve_samples else 1
  right_curve_weight = left_curve_samples / right_curve_samples if right_curve_samples > left_curve_samples else 1
  new_data = []
  for line in _data:
    if line['torque'] > threshold and random_chance(left_curve_weight * 100):
      new_data.append(line)
    elif line['torque'] < -threshold and random_chance(right_curve_weight * 100):
      new_data.append(line)
    elif abs(line['torque']) <= threshold:
      new_data.append(line)
  return new_data

def plot_distributions(_data, idx=0):
  # key_lists = {k: [line[k] for line in _data] for k in STATS_KEYS}
  key_lists = {}
  for stat_k, data_keys in STATS_KEYS.items():
    key_lists[stat_k] = []
    for data_k in data_keys:  # handles if stats key has multiple data keys in same category
      key_lists[stat_k] += [line[data_k] for line in _data]

  # angle_errors = [abs(line['steering_angle'] - line['fut_steering_angle']) for line in _data]
  # key_lists['angle_errors'] = angle_errors

  for key in key_lists:
    bins = 200  # int(interp(len(key_lists[key]), [800000, 100000], [50, 200]))
    kde = len(key_lists[key]) < 400000

    plt.clf()
    sns.distplot(key_lists[key], bins=bins, kde=kde)
    plt.savefig('plots/{} dist.{}.png'.format(key, idx))
    # if key == 'torque':
    #   raise Exception


def get_stats(_data):
  @add_objprint
  class Stat:
    def __init__(self, name, mean, std, mn, mx):
      self.name = name
      self.mean = mean
      self.std = std
      self.scale = [mn, mx]
      cut_off_multiplier = stds if name != 'torque' else stds * 2
      self.cut_off = [mean - std * cut_off_multiplier, mean + std * cut_off_multiplier]

  stds = 3
  key_lists = {}
  for stat_k, data_keys in STATS_KEYS.items():
    key_lists[stat_k] = []
    for data_k in data_keys:  # handles if stats key has multiple data keys in same category
      key_lists[stat_k] += [line[data_k] for line in _data]

  stats = {k: Stat(k, np.mean(key_lists[k]),
                   np.std(key_lists[k]),
                   min(key_lists[k]),
                   max(key_lists[k])) for k in STATS_KEYS}
  return stats


def remove_outliers(_flattened):  # calculate current mean and std to filter, then return the newly updated mean and std
  stats = get_stats(_flattened)
  print(stats['angle'].mean, stats['angle'].std)

  print('Data cut offs: {}'.format({k: stats[k].cut_off for k in stats}))

  new_data = []
  for line in _flattened:
    keep = []
    for stat_k, data_keys in STATS_KEYS.items():
      for data_k in data_keys:
        keep.append(stats[stat_k].cut_off[0] < line[data_k] < stats[stat_k].cut_off[1])

    keep = all(keep)
    if keep:  # if sample falls within standard deviation * 3
      new_data.append(line)

  return new_data


class SyntheticDataGenerator:
  def __init__(self, _data, _stats):
    self.data = _data
    self.torque_range = [_stats['torque'].std, max(np.abs(_stats['torque'].scale)) * 1.5]
    print(f'Torque range: {self.torque_range}')

    self.max_idx = len(self.data) - 1
    self.keys = ['fut_steering_angle', 'steering_angle', 'fut_steering_rate', 'steering_rate', 'v_ego', 'a_ego']
    self.idxs_needed = len(self.keys)
    self.pid = LatControlPF()

  def generate_many(self, n):
    return [self.generate_one() for _ in range(n)]

  def generate_one(self):
    def _gen():
      idxs = [random.randint(0, self.max_idx) for _ in range(self.idxs_needed)]
      _sample = {}
      for key, idx in zip(self.keys, idxs):
        _sample[key] = self.data[idx][key]  # todo: maybe randomly transform them by a small number?

      _sample['angle_error'] = abs(_sample['steering_angle'] - _sample['fut_steering_angle'])
      _sample['torque'] = self.pid.update(_sample['fut_steering_angle'], _sample['steering_angle'], _sample['v_ego']) * TORQUE_SCALE
      return _sample

    sample = _gen()
    while abs(sample['torque']) > self.torque_range[1] or abs(sample['torque']) < self.torque_range[0] or sample['angle_error'] < 3 or sample['angle_error'] > 25:
      sample = _gen()
    return sample
    # this was fairly accurate, but the above will automatically change with the data (no manual tuning required)
    # choice = random.choice([0, 1])  # 0 is actual angle std, 1 is std / 8 to replicate much more samples near 0
    # return (np.random.normal(0, angles_std) if choice == 0 else
    #         np.random.normal(0, angles_std / 8))


def load_data(fn='data', to_normalize=False, plot_dists=False):  # filters and processes raw pickle data from rlogs
  data_sequences = load_processed(fn)


  # for sec in data:
  #   print('len: {}'.format(len(sec)))
  #   print('num. 0 steering_rate: {}'.format(len([line for line in sec if line['steering_rate'] == 0])))
  #   print()

  # there is a delay between sending torque and reaching the angle
  # this adds future steering angle and rate data to each sample, which we will use to train on as inputs
  # data for model: what current torque (output) gets us to the future (input)
  # this makes more sense than training on desired angle from lateral planner since humans don't always follow what the mpc would predict in any given situation
  data_sequences = offset_torque(data_sequences)

  for seq in data_sequences:  # add angle error
    for line in seq:
      line['angle_error'] = abs(line['steering_angle'] - line['fut_steering_angle'])

  # filter data
  data_sequences = filter_data(data_sequences)  # returns filtered sequences

  # flatten into 1d list of dictionary samples
  flat_samples = [i.copy() for j in data_sequences for i in j]  # make a copy of each list sample so any changes don't affect data_sequences

  print('Flat samples: {}'.format(len(flat_samples)))

  if plot_dists:
    plot_distributions(flat_samples)  # this takes a while

  # Remove outliers
  filtered_data = remove_outliers(flat_samples)  # returns stats about filtered data
  print('Removed outliers: {} samples'.format(len(filtered_data)))

  if plot_dists:
    plot_distributions(filtered_data, 1)

  # Remove inliers  # too many samples with angle at 0 degrees compared to curve data
  filtered_data_new = []
  for line in filtered_data:
    # if abs(line['torque']) > 500:
    #   filtered_data_new.append(line)
    # elif random_chance(interp(line['torque'], [-471, -105, 95, 194, 494], [100, 40 / 2, 35 / 2, 45 / 2, 100])):
    #   filtered_data_new.append(line)

    if abs(line['steering_angle']) > 90:
      filtered_data_new.append(line)
    elif random_chance(interp(abs(line['steering_angle']), [0, 45, 90], [25, 75, 100])):
      filtered_data_new.append(line)
  data = filtered_data_new
  del filtered_data_new

  data = even_out_torque(data)  # there's more left angled samples than right for some reason


  print('Removed inliers: {} samples'.format(len(data)))

  data_stats = get_stats(data)  # get stats about final filtered data

  if plot_dists:
    plot_distributions(data, 2)

  print(f'Angle mean, std: {data_stats["angle"].mean, data_stats["angle"].std}')

  data_generator = SyntheticDataGenerator(data, data_stats)
  ADD_SYNTHETIC_SAMPLES = True  # fixme, this affects mean and std, but not min/max for normalizing
  if ADD_SYNTHETIC_SAMPLES:

    n_synthetic_samples = round(len(data) / 20)
    print('There are currently {} real samples'.format(len(data)))
    print('Adding {} synthetic samples...'.format(n_synthetic_samples), flush=True)
    data += data_generator.generate_many(n_synthetic_samples)

    print('Real and synthetic samples: {}'.format(len(data)))

    # todo: we could just update all stats, not sure why we're only doing torque
    # todo: do we want the stats to represent the real data only, or data we're going to train on (real and synthetic)?
    torque = [line['torque'] for line in data]
    data_stats['torque'].mean = np.mean(torque)
    data_stats['torque'].std = np.std(torque)
    data_stats['torque'].scale = [min(torque), max(torque)]  # scale is most important

    print('Added synthetic data: {} samples'.format(len(data)))

    if plot_dists:
      plot_distributions(data, 3)  # this takes a while

  # Normalize data
  if to_normalize:
    data = [normalize_sample(line, data_stats, to_normalize) for line in data]

  # Return flattened samples, original sequences of data (filtered), and stats about filtered_data
  return data, data_sequences, data_stats, data_generator

  # filtered_data = []  # todo: check for disengagement (or engagement if disengaged) or user override in future
  # for sec in data:  # remove samples where we're braking in the future but not now
  #   new_sec = []
  #   for idx, line in enumerate(sec):
  #     accel_delay = get_accel_delay(line['v_ego'])  # interpolate accel delay from speed
  #     if idx + accel_delay < len(sec):
  #       if line['brake_pressed'] is sec[idx + accel_delay]['brake_pressed']:
  #         new_sec.append(line)
  #   if len(new_sec) > 0:
  #     filtered_data.append(new_sec)
  # data = filtered_data
  # del filtered_data



if __name__ == "__main__":
  data, data_sequences, data_stats, data_generator = load_data(plot_dists=True)
  # plt.plot([line['steering_angle'] for line in data_sequences[3]])
  #
  # raise Exception
  del data_sequences

  plt.clf()
  print(f'stats rate mean, std: {data_stats["torque"].mean, data_stats["rate"].std}')

  angles = [l['torque'] for l in data]
  print(f'real torque mean, std: {np.mean(angles), np.std(angles)}')
  sns.distplot(angles, label='torque in data', bins=200)

  angles_std = np.std(angles)

  # generated_angles = [np.random.normal(0, angles_std * 1.4) for _ in range(len(angles))]
  # generated_angles += [np.random.normal(0, angles_std/12) for _ in range(int(len(angles)))]
  # generated_angles = [generate_syn_angle() for _ in range(len(angles) * 2)]
  generated_angles = [data_generator.generate_one()['torque'] for _ in range(len(angles))]
  sns.distplot(generated_angles, label='generated torque', bins=200)

  plt.legend()



# if __name__ == "__main__":
#   speed_range = [10.778, 13.16]
#   data, data_sequences, data_stats = load_data()
#   data = data_sequences[-1]
#
#   for idx, line in enumerate(data):
#     # factor = [0.1, 0.05]
#     # factor = [0.0666, 0.0333]
#     # factor = [0.08, 0.04]
#     factor = [0.05, 0.025]
#     line['steering_rate_fraction'] = line['steering_rate_fraction'] * factor[0] + factor[1]
#     past = 5
#     if idx < past:
#       line['steering_rate_calculated'] = 0
#     else:
#       calculated = line['steering_angle'] - data[idx - past]['steering_angle']
#       line['steering_rate_calculated'] = calculated * (100 / past) if calculated != 0 else 0
#
#   print(len(data_sequences))
#   plt.title('{} factor, {} offset'.format(factor[0], factor[1]))
#   fraction = [line['steering_rate_fraction'] for line in data]
#   # plt.plot([line['steering_rate_can'] for line in data], label='with fraction')
#   # plt.plot([line['steering_rate_calculated'] for line in data], label='calculated ({} second)'.format(past / 100))
#   # plt.plot([line['steering_rate'] for line in data], label='w/o fraction')
#   plt.plot([line['steering_rate'] + line['steering_rate_fraction'] for line in data], label='fraction')
#   # plt.plot(fraction, label='fraction')
#   print(f'min max: {min(fraction)}, {max(fraction)}')
#   plt.legend()
#
#   raise Exception
#
#   # # data = [l for l in data if not l['engaged']]
#   # data = [l for l in data if speed_range[0] <= l['v_ego'] <= speed_range[1]]
#   #
#   # plt.figure()
#   # plt.plot([l['torque_eps'] for l in data], label='eps')
#   # plt.plot([l['torque_driver'] for l in data], label='driver')
#   # plt.legend()
#   #
#   # plt.figure()
#   # sns.distplot([l['v_ego'] for l in data], bins=200)
#   #
#   # plt.figure()
#   # angles = [abs(l['fut_steering_angle']) for l in data]
#   # torque = [abs(l['torque']) for l in data]
#   # x = np.linspace(0, max(angles), 100)
#   # plt.plot(x, [feedforward(_x, np.mean(speed_range)) * 0.00006908923778520113 * 1500 for _x in x])
#   # plt.scatter(angles, torque, s=0.5)
#   # plt.legend()

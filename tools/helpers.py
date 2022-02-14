import random
# from tensorflow import keras
from common.numpy_fast import interp
from torque_model.models.ff.feedforward_model import predict as feedforward_predict

from selfdrive.config import Conversions as CV

TORQUE_SCALE = 1500  # toyota
# TORQUE_SCALE = 300  # volks
STATS_KEYS = {'angle': ['fut_steering_angle', 'steering_angle'], 'rate': ['fut_steering_rate', 'steering_rate'], 'speed': ['v_ego'], 'accel': ['a_ego'], 'torque': ['torque'], 'angle_error': ['angle_error']}  # this renames keys to shorter names to access later quicker
REVERSED_STATS_KEYS = {}
for stat_k, data_keys in STATS_KEYS.items():
  for data_k in data_keys:
    REVERSED_STATS_KEYS[data_k] = stat_k


MODEL_INPUTS = ['fut_steering_angle', 'steering_angle', 'fut_steering_rate', 'steering_rate', 'v_ego', 'a_ego']
# inputs = ['fut_steering_angle', 'steering_angle', 'v_ego']


def unnormalize_sample(_sample, _stats):
  _sample = _sample.copy()
  for inp in MODEL_INPUTS:
    _sample[inp] = interp(_sample[inp], [-1, 1], _stats[REVERSED_STATS_KEYS[inp]].scale)
  return _sample


def normalize_sample(_sample, _stats, _normalize):
  _sample = _sample.copy()
  if not _normalize:
    return _sample

  for inp in MODEL_INPUTS:
    _sample[inp] = interp(_sample[inp], _stats[REVERSED_STATS_KEYS[inp]].scale, [-1, 1])
  return _sample


def normalize_value(_v, _type, _stats, _normalize):
  return interp(_v, _stats[_type].scale, [-1, 1]) if _normalize else _v


def feedforward(angle, speed):
  steer_feedforward = float(angle)  # offset does not contribute to resistive torque
  _c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
  steer_feedforward *= _c1 * speed ** 2 + _c2 * speed + _c3
  # steer_feedforward *= speed ** 2
  return steer_feedforward


# feedforward_model = keras.models.load_model('models/feedforward_model.h5', custom_objects={'LeakyReLU': keras.layers.LeakyReLU})


def model_feedforward(angle, speed):
  # return float(feedforward_model.predict_on_batch(np.array([[angle, angle, 0, speed]]))[0][0])
  return float(feedforward_predict([angle, angle, 0, speed])[0])


class LatControlPF:
  def __init__(self):
    self.k_f = 0.00008
    # self.k_f = 0.00003
    self.speed = 0

  @property
  def k_p(self):
    return interp(self.speed, [20 * CV.MPH_TO_MS, 70 * CV.MPH_TO_MS], [.1, .2])

  def update(self, setpoint, measurement, speed):
    self.speed = speed
    f = feedforward(setpoint, speed)
    # f = model_feedforward(setpoint, speed)

    error = setpoint - measurement

    p = error * self.k_p
    f = f * self.k_f

    return p + f  # multiply by 1500 to get torque units
    # return np.clip(p + steer_feedforward, -1, 1)  # multiply by 1500 to get torque units


def random_chance(percent: int):
  return percent == 0 or random.randint(0, 100) < percent

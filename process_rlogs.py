from opendbc.can.parser import CANParser
from tools.lib.logreader import MultiLogIterator
from cereal import car
from common.basedir import BASEDIR
from selfdrive.car.toyota.values import CAR as TOYOTA_CAR, DBC as TOYOTA_DBC
from tqdm import tqdm   # type: ignore
import pickle
import os

os.chdir(os.path.join(BASEDIR, 'torque_model'))

DT_CTRL = 0.01
MIN_SAMPLES = 5 / DT_CTRL  # seconds to frames
TRAIN_CARS = [TOYOTA_CAR.CAMRYH_TSS2]


def load_and_process_rlogs(lrs, file_name):
  data = [[]]

  for lr in lrs:
    CS = None
    # engaged, v_ego, a_ego, gear_shifter = None, None, None, None
    engaged = None
    last_engaged = None

    last_time = 0
    can_updated = False

    signals = [
      ("STEER_REQUEST", "STEERING_LKA", 0),
      ("STEER_TORQUE_CMD", "STEERING_LKA", 0),
      ("STEER_TORQUE_DRIVER", "STEER_TORQUE_SENSOR", 0),
      ("STEER_TORQUE_EPS", "STEER_TORQUE_SENSOR", 0),
    ]
    cp = None
    car_fingerprint = None

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    for msg in tqdm(all_msgs):
      if msg.which() == 'carState':
        CS = msg.carState

        last_engaged = bool(engaged)
        engaged = msg.carState.cruiseState.enabled
      # elif msg.which() == 'carControl':  # todo: maybe get eps torque
      #   apply_accel = msg.carControl.actuators.gas - msg.carControl.actuators.brake
      elif msg.which() == 'carParams':
        car_fingerprint = msg.carParams.carFingerprint
        if car_fingerprint not in TRAIN_CARS:
          raise Exception("Car not supported for training at this time: {}".format(car_fingerprint))
        if cp is None:
          cp = CANParser(TOYOTA_DBC[car_fingerprint]['pt'], signals, enforce_checks=False)

      if cp is None:  # no carParams msg yet
        continue

      if msg.which() in ['can', 'sendcan']:
        cp_updated = cp.update_string(msg.as_builder().to_bytes())  # usually all can signals are updated so we don't need to iterate through the updated list
        for u in cp_updated:
          if u == 608:  # STEER_TORQUE_SENSOR
            can_updated = True

      if msg.which() != 'can':  # only store when can is updated
        continue

      # wait for first carState msg and CAN is updated
      if not can_updated or CS is None:
        continue

      # TODO: this should be adjusted by make/model
      torque_cmd = cp.vl['STEERING_LKA']['STEER_TORQUE_CMD']
      torque_eps = cp.vl['STEER_TORQUE_SENSOR']['STEER_TORQUE_EPS']
      torque_driver = cp.vl['STEER_TORQUE_SENSOR']['STEER_TORQUE_DRIVER']
      steer_req = cp.vl['STEERING_LKA']['STEER_REQUEST'] == 1

      if abs(msg.logMonoTime - last_time) * 1e-9 > 1 / 20:
        print('TIME BREAK!')
        print(abs(msg.logMonoTime - last_time) * 1e-9)

      # gather data if user driving, or engaged and not user override
      should_gather = not engaged or (engaged and steer_req and not CS.steeringPressed)

      # creates uninterrupted sections of engaged data
      if (should_gather and CS.gearShifter == car.CarState.GearShifter.drive and engaged == last_engaged and
              abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20):  # also split if there's a break in time
        data[-1].append({'v_ego': CS.vEgo, 'a_ego': CS.aEgo, 'steering_angle': CS.steeringAngleDeg, 'steering_rate': CS.steeringRateDeg,
                         'engaged': engaged, 'torque_cmd': torque_cmd, 'torque_eps': torque_eps, 'torque_driver': torque_driver,
                         'car_fingerprint': car_fingerprint, 'time': msg.logMonoTime * 1e-9})
      elif len(data[-1]):  # if last list has items in it, append new empty section
        data.append([])

      last_time = msg.logMonoTime

  del all_msgs

  data = [seq for seq in data if len(seq) > MIN_SAMPLES]  # long enough sections

  seq_lens = [len(seq) for seq in data]
  print('Max seq. len: {} ({} s)'.format(max(seq_lens), max(seq_lens) * DT_CTRL))
  print('Sum seq. len: {} ({} s)'.format(sum(seq_lens), sum(seq_lens) * DT_CTRL))
  print('Sequences: {}'.format([len(seq) for seq in data]))

  with open(file_name, 'wb') as f:  # now dump
    pickle.dump(data, f)
  return data


if __name__ == "__main__":
  rlog_dir = 'rlogs'
  route_dirs = [f for f in os.listdir(os.path.join(rlog_dir)) if '.ini' not in f and f != 'exclude']
  route_files = []
  for route in route_dirs:
    route_files.append([])
    for segment in os.listdir(os.path.join(rlog_dir, route)):
      if segment == 'exclude' or '.ini' in segment:
        continue
      route_files[-1].append(os.path.join(rlog_dir, route, segment))

  lrs = [MultiLogIterator(rd, wraparound=False) for rd in route_files]
  data = load_and_process_rlogs(lrs, file_name='data')

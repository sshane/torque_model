import os
import urllib
import shutil
from common.basedir import BASEDIR
from tools.lib.route import Route

rlog_path = os.path.join(BASEDIR, "torque_model", "rlogs")

if __name__ == "__main__":
  route = input("Enter route name: ")
  route = Route(route)
  route_name = route.route_name.replace("|", "_")

  route_path = os.path.join(rlog_path, route_name)
  print("Saving into {}".format(route_path))

  os.makedirs(route_path, exist_ok=True)
  os.makedirs(os.path.join(route_path, "exclude"), exist_ok=True)
  os.makedirs(os.path.join(route_path, ".tmp"), exist_ok=True)
  for idx, segment in enumerate(route.segments):
    segment_fn = segment.name.replace("|", "_") + "--rlog.bz2"
    tmp_log_path = os.path.join(route_path, ".tmp", segment_fn)
    local_log_path = os.path.join(route_path, segment_fn)

    # skip if already downloaded fully
    if os.path.exists(local_log_path) and not os.path.exists(tmp_log_path):
      print("Skipping already downloaded file: {}".format(segment_fn))
      continue

    print("Downloading ({} of {}): {}".format(idx + 1, len(route.segments), segment_fn))
    urllib.request.urlretrieve(segment.log_path, tmp_log_path)

    # move downloaded file to final location
    shutil.move(tmp_log_path, local_log_path)

  shutil.rmtree(os.path.join(route_path, ".tmp"))

  print("Finished downloading route to {}".format(route_path))

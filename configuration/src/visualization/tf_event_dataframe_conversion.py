import pandas as pd
import os
import sys
import logging
import glob
import tensorflow as tf
# import matplotlib.pyplot as plt
from argparse import ArgumentParser

def configure_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_dir', "--expdir", type=str)
    parser.add_argument('--event_dir', "--evdir", type=str, default="agent/summaries")
    return parser.parse_args(sys.argv[1:])

args = configure_args()
exp_dir = args.exp_dir
summaries_subdir = args.event_dir

summaries_dir = os.path.join(exp_dir, summaries_subdir)

dataframe_filename = os.path.join(exp_dir, "agent/dataframe")
# data as a dict of map(step, map(tag): value)
data = {}
event_files = [os.path.join(summaries_dir, f) for f in os.listdir(summaries_dir)]

print(f"{len(event_files)} event files found")

for f in event_files:
  # summary_iterator deprecated from tf 2.0
  for e in tf.compat.v1.train.summary_iterator(f):
    print("Event", e, e.summary.value, e.step)
    for value in e.summary.value:
      if data.get(e.step, None) is None:
        data[e.step] = {}
      data[e.step][value.tag] = value.simple_value
      print("Value", value.tag, value.simple_value)

# write to dataframe
df = pd.DataFrame.from_dict(data, orient='index')

# save dataframe
print(f"Saving Dataframe to file {dataframe_filename}")
df.to_pickle(dataframe_filename)

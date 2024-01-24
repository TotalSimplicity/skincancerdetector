import os
import csv

image_dir = '/home/leo/Documents/ScienceFair2023/images'

csv_file = '/home/leo/Documents/ScienceFair2023/HAM10000_metadata.csv'

labels = {}

with open(csv_file) as f:
  reader = csv.reader(f)
  next(reader)  # skip the header row
  for row in reader:
    labels[row[1]] = row[2]

for file in os.listdir(image_dir):
  image_id = file.split('.')[0]
  label = labels[image_id]

  label_dir = os.path.join(image_dir, label)
  if not os.path.exists(label_dir):
    os.makedirs(label_dir)

  src = os.path.join(image_dir, file)
  dst = os.path.join(label_dir, file)
  os.rename(src, dst)

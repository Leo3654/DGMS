import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to json files')
args = parser.parse_args()

error_counts = {}

for filename in os.listdir(args.path):
    if filename.endswith('.json'):
        with open(os.path.join(args.path, filename)) as f:
            data = json.load(f)
            error = data.get('error')
            if error:
                error_counts[error] = error_counts.get(error, 0) + 1

print(error_counts)
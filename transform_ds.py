import tensorflow as tf
import numpy as np
import sys
import json
from pprint import pprint
from google.cloud import storage
from google.oauth2 import service_account
service_account.Credentials.service_account_info = json.load(open('/content/tmp/adc.json', 'r'))
creds = service_account.Credentials.from_service_account_file('/content/tmp/adc.json')
GOOGLE_APPLICATION_CREDENTIALS = json.load(open('/content/tmp/adc.json', 'r'))
storage_client = storage.Client.from_service_account_json('/content/tmp/adc.json')

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Transform JSONL dataset to CSV')
parser.add_argument(
    '--input_fn',
    dest='input_fn',
    type=str,
    help='Path to a JSONL containing metadata',
)
parser.add_argument(
    '--output_fn',
    dest='output_fn',
    type=str,
    help='base output filename /path/exp1 will yield /path/exp1_{train|test|val}.csv',
)
parser.add_argument(
    '--label',
    dest='label_field',
    type=str,
    default='label',
    help='Key for label, if not label',
)
parser.add_argument(
    '--text',
    dest='text_field',
    type=str,
    default='text',
    help='Key for Target Text if not the key is not text',
)
parser.add_argument(
    '--format',
    dest='format',
    type=str,
    default='jsonl',
    help='Output file format',
)

args = parser.parse_args()

with tf.io.gfile.GFile(args.input_fn, 'r+') as f:
    items = [json.loads(l) for i, l in enumerate(f)]
    total_items = len(items)
    print('Loaded {} Items'.format(total_items))

#assert args.text_field in items
#assert args.label_field in items

train_fn = tf.io.gfile.GFile(('{}_train.{}'.format(args.output_fn, args.format)), 'w+')
test_fn = tf.io.gfile.GFile(('{}_test.{}'.format(args.output_fn, args.format)), 'w+')
val_fn = tf.io.gfile.GFile(('{}_val.{}'.format(args.output_fn, args.format)), 'w+')

if args.format == 'csv':
    train_fn.write('label,text\n')
    test_fn.write('label,text\n')
    val_fn.write('label,text\n')



for i, item in enumerate(tqdm(items)):
    if args.format == 'jsonl':
        result = {
            'label': item[args.label_field],
            'text': item[args.text_field]
        }
        if i % 8 == 0:
            val_fn.write(json.dumps(result) + '\n')
        elif i % 12 == 0:
            test_fn.write(json.dumps(result) + '\n')
        else:
            train_fn.write(json.dumps(result) + '\n')

    elif args.format == 'csv':
        label = item[args.label_field]
        text = item[args.text_field]
        result = '{},{}\n'.format(label, text)

        if i % 8 == 0:
            val_fn.write(result)
        elif i % 12 == 0:
            test_fn.write(result)
        else:
            train_fn.write(result)

train_fn.close()
test_fn.close()
val_fn.close()







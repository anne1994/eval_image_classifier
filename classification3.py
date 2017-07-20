#!/usr/bin/env python

from __future__ import print_function
import sys
#sys.path.append('../../tensorflow/models/slim/') # add slim to PYTHONPATH
import tensorflow as tf
#from sklearn.metrics import confusion_matrix
import re
import os.path



tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of classes.')
tf.app.flags.DEFINE_string('infile',None, 'Image file, one image per line.')
tf.app.flags.DEFINE_boolean('tfrecord',False, 'Input file is formatted as TFRecord.')
tf.app.flags.DEFINE_string('outfile',None, 'Output file for prediction probabilities.')
tf.app.flags.DEFINE_string('model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', '/data/tmp/prescan-models-train/inception_v3','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', None, 'Eval image size.')
tf.app.flags.DEFINE_string('model_dir','/data/models/slim', ' model dir')
FLAGS = tf.app.flags.FLAGS

from six.moves import urllib
import numpy as np
import os


from datasets import imagenet
from nets import inception_v3
#from nets import resnet_v1
from nets import inception_utils
from nets import resnet_utils
from preprocessing import inception_preprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               uid_lookup_path=None):
   # if not label_lookup_path:
     # label_lookup_path = os.path.join(
         # FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'labels.txt')
    self.node_lookup = self.load(uid_lookup_path)

  def load(self,uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    #if not tf.gfile.Exists(label_lookup_path):
     # tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    #proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    #uid_to_human = {}
    #p = re.compile(r'[n\d]*[ \S,]*')
    #for line in proto_as_ascii_lines:
      #parsed_items = p.findall(line)
      #uid = parsed_items[0]
      #human_string = parsed_items[2]
      #uid_to_human[uid] = human_string

    #Loads mapping from string UID to integer node ID.
    #node_id_to_uid = {}
    #proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    #for line in proto_as_ascii:
      #if line.startswith('  target_class:'):
        #target_class = int(line.split(': ')[1])
      #if line.startswith('  target_class_string:'):
        #target_class_string = line.split(': ')[1]
        #node_id_to_uid[target_class] = target_class_string[1:-2]

    #Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(uid_lookup_path).readlines()
    for line in proto_as_ascii:
        target_class = int(line.split(':')[0])
       # print('target class', target_class)
        target_class_string = line.split(':')[1]
       # print('target class string', target_class_string)
        node_id_to_uid[target_class] = target_class_string[0:-1]
    print('nodeid to uid ', node_id_to_uid)

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      #print('key', key)
      #print('val', val)
      if key not in node_id_to_uid:
        tf.logging.fatal('Failed to locate: %s', val)
      name = node_id_to_uid[key]
      #print('name',name)
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]



model_name_to_variables = {'inception_v3':'InceptionV3'}

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
eval_image_size = FLAGS.eval_image_size

if FLAGS.tfrecord:
  fls = tf.python_io.tf_record_iterator(path=FLAGS.infile)
  print('fls',fls)
else:
  fls = [s.strip() for s in open(FLAGS.infile)]
  print("fls is ", fls)
model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if FLAGS.tfrecord:
  tf.logging.warn('Image name is not available in TFRecord file.')

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path
print("checkpoint path successful")

image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) ## To process corrupted image files

image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)

if FLAGS.eval_image_size is None:
  eval_image_size = network_fn.default_image_size

processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

processed_images  = tf.expand_dims(processed_image, 0) # Or tf.reshape(processed_image, (1, eval_image_size, eval_image_size, 3))

logits, _ = network_fn(processed_images)
print("logits",logits)

probabilities = tf.nn.softmax(logits)
print("probabilites",probabilities)

init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))

sess = tf.Session()
init_fn(sess)
print("session started")
fout = sys.stdout
if FLAGS.outfile is not None:
  fout = open(FLAGS.outfile, 'w')
h = ['image']
h.extend(['class%s  ' % i for i in range(FLAGS.num_classes)])
h.append('predicted_class')
h.append('predicted_class_name')
print('\t \t '.join(h), file=fout)

#print('predicted class')


for fl in fls:
  image_name = None

  try:
    if FLAGS.tfrecord is False:
      y = tf.gfile.FastGFile(fl).read() # You can also use x = open(fl).read()
      #print('x is in if loop',y)
      image_name = os.path.basename(fl)
    else:
      example = tf.train.Example()
      example.ParseFromString(fl)
      #print('else loop')  
      # Note: The key of example.features.feature depends on how you generate tfrecord.
      y = example.features.feature['image/encoded'].bytes_list.value[0] # retrieve image string

      image_name = 'TFRecord'

    probs = sess.run(probabilities, feed_dict={image_string:y})
    
    #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})

  except Exception as e:
    print('exception')
    tf.logging.warn('Cannot process image file %s' % fl)
    continue
  node_lookup = NodeLookup()
  probs = probs[0, 0:]
  a = [image_name]
  a.extend(probs)
  prediction=np.argmax(probs)
  a.append(prediction)
  a.append(node_lookup.id_to_string(prediction))
  print('\t \t  '.join([str(e) for e in a]), file=fout)
  print('image name is ' , a)
  top_k = [i[0] for i in sorted(enumerate(-probs), key = lambda x:x[1])]
  print(top_k)
  for node_id in top_k:
	human_string = node_lookup.id_to_string(node_id)
        #print('human_string is', human_string)
	score=probs[node_id]
	print('%s (score = %.5f)' % (human_string, score))

   #a.append(node_lookup.id_to_string(prediction)
   #print('\t'.join([str(e) for e in a]), file=fout)
  print('######################################################') 
sess.close()
fout.close()







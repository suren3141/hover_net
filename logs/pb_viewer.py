# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.platform import gfile

# from tensorflow.compat.v2.io import gfile
# from tensorflow.compat.v1 import GraphDef   # -> instead of tf.GraphDef()
# tf.compat.v2.io.gfile.GFile()   # -> instead of tf.gfile.GFile()

import os
from google.protobuf import text_format

log_dir = "./logs/ResNet18"
pbtxt_file = "projector_config.pbtxt"

with open(os.path.join(log_dir, pbtxt_file)) as f:
    text_graph = f.read()
graph_def = text_format.Parse(text_graph, tf.GraphDef())
tf.io.write_graph(graph_def, )
tf.train.write_graph(graph_def, log_dir, 'graph.pb', as_text=False)

with tf.Session() as sess:
    model_filename =os.path.join(log_dir, "graph.pb")
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='/logs/tests/1/'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
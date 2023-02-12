# from tensorflow.python._pywrap_tensorflow_internal import *
# from tensorflow.python.framework import _pywrap_python_op_gen
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

aaa = tf.constant([1,1,1,1],name="kkk")
bbb = tf.constant([2,2,2,2],name="sss")
add = tf.math.add(aaa,bbb,name="zzz")
square = tf.math.square(add,name="yyy")
add_n = tf.math.add_n([aaa,bbb,add],name='addn')
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
node2 = graph_def.node[2]
nodes = [n for n in tf.get_default_graph().as_graph_def().node]
# for node in nodes:
#     print(node)
with tf.Session() as sess:
    eee = [op for op in sess.graph.get_operations()]
    for op in eee:
        print(op.op_def)
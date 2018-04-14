import tensorflow as tf
import numpy as np


def test_embedding_look_up():
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None])

    #embedding = tf.get_variable('test', shape=[5,5])
    embedding = tf.Variable(np.identity(5, dtype=np.int32))
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("embeding:\n", embedding.eval())
    result = sess.run(input_embedding, feed_dict={input_ids:[[[1, 2], [1, 2]], [[2, 1], [2, 1]], [[3, 3], [3, 3]]]})
    print("结果：\n", result)
    print(result.shape)
test_embedding_look_up()
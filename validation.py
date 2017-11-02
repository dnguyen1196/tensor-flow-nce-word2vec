import tensorflow as tf

tf.reset_default_graph()

final_embeddings = tf.get_variable("final_embeddings")

saver = tf.train.Saver()
saved_model_path = "/home/duc/Documents/homework/Research/Tensorflow/Codes/saved_variables/model.ckpt"

with tf.Session() as session:
    saver.restore(session, saved_model_path)
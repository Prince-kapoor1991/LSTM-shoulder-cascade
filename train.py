import tensorflow as tf
import train_utils, googlenet_load
import os
import json
import random
import time
import string
import datetime
import argparse
from scipy import misc
import numpy as np
import tensorflow.contrib.slim as slim
import cv2 as cv
try:
    from tensorflow.models.rnn import rnn_cell
except ImportError:
    rnn_cell=tf.nn.rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

random.seed(0)
np.random.seed(0)

save_dir=os.getcwd()+'/output'
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir','/home/vivalab/opencv_examples/train.json',"""Directory for training set""")
tf.app.flags.DEFINE_string('test_dir','/home/vivalab/opencv_examples/valid.json',"""directory for validation set""")
tf.app.flags.DEFINE_string('save_dir',save_dir,"""directory to save the values""")

tf.app.flags.DEFINE_integer('rnn_len',1,"""number of RNN network""")
tf.app.flags.DEFINE_integer('num_classes',2,"""numer of classes for the network""")
tf.app.flags.DEFINE_integer('image_height',480,"""Height of image""")
tf.app.flags.DEFINE_integer('image_width',640,"""width of image""")
tf.app.flags.DEFINE_integer('grid_height',15,"""Height of grid""")
tf.app.flags.DEFINE_integer('grid_width',20,"""width of grid""")
tf.app.flags.DEFINE_integer('batch_size',1,"""batch size""")

tf.app.flags.DEFINE_float('learning_rate',0.001,"""learning for training""")
tf.app.flags.DEFINE_integer('learning_rate_step',50000,"""number of iterations after which learning rate would decrease""")
tf.app.flags.DEFINE_integer('later_feat_channels',832,"""channels for later feat""")
tf.app.flags.DEFINE_integer('lstm_size',500,"""size of lstm input""")
tf.app.flags.DEFINE_integer('early_feat_channels',256,"""channels for early feat""")
tf.app.flags.DEFINE_integer('avg_pool_size',5,"""size for average pooling after inception""")
tf.app.flags.DEFINE_float('clip_norm',1.0,"""clip normalization""")

tf.app.flags.DEFINE_integer('display_iter',50,"""iteration at which the result is displayed""")
tf.app.flags.DEFINE_integer('save_iter',10000,"""iteration at which the checkpoint is saved""")

def build_overfeat_inner(lstm_input):
    '''
    build simple overfeat decoder
    '''
    if FLAGS.rnn_len > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[FLAGS.later_feat_channels, FLAGS.lstm_size])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs

def model(x, phase, reuse):
    grid_size=FLAGS.grid_width*FLAGS.grid_height
    outer_size=grid_size*FLAGS.batch_size
    input_mean=117
    x -=input_mean
    cnn,early_feat, _ = googlenet_load.model(x,reuse)
    early_feat_channels = FLAGS.early_feat_channels
    early_feat = early_feat[:, :, :, :early_feat_channels]

    if FLAGS.avg_pool_size > 1:
        pool_size = FLAGS.avg_pool_size
        cnn1 = cnn[:, :, :, :700]
        cnn2 = cnn[:, :, :, 700:]
        cnn2 = tf.nn.avg_pool(cnn2, ksize=[1, pool_size, pool_size, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
        cnn = tf.concat(3, [cnn1, cnn2])

    cnn = tf.reshape(cnn,[FLAGS.batch_size * FLAGS.grid_width * FLAGS.grid_height, FLAGS.later_feat_channels])
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('decoder', reuse=reuse, initializer=initializer):
        scale_down = 0.01
        lstm_input = tf.reshape(cnn * scale_down, (FLAGS.batch_size * grid_size, FLAGS.later_feat_channels))
        lstm_outputs = build_overfeat_inner(lstm_input)

        pred_boxes = []
        pred_logits = []
        for k in range(FLAGS.rnn_len):
            output = lstm_outputs[k]
            if phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(FLAGS.lstm_size, 4))
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(FLAGS.lstm_size, FLAGS.num_classes))

            pred_boxes_step = tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4])

            pred_boxes.append(pred_boxes_step)
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                         [outer_size, 1, FLAGS.num_classes]))

        pred_boxes = tf.concat(1, pred_boxes)
        pred_logits = tf.concat(1, pred_logits)
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * FLAGS.rnn_len, FLAGS.num_classes])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)

        pred_confidences = tf.reshape(pred_confidences_squash,[outer_size, FLAGS.rnn_len, FLAGS.num_classes])
        x=pred_boxes
    return pred_boxes, pred_logits, pred_confidences,x

def losses(x,phase,boxes,flags):
    grid_size=FLAGS.grid_width*FLAGS.grid_height
    head_weights=[1.0, 0.1]
    outer_size=grid_size*FLAGS.batch_size
    reuse={'train':None, 'test':True}[phase]
    pred_boxes,pred_logits,pred_confidences,x=model(x,phase,reuse)
    with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, FLAGS.rnn_len, 4])
        outer_flags = tf.cast(tf.reshape(flags, [outer_size, FLAGS.rnn_len]), 'int32')
        classes = tf.reshape(flags, (outer_size, 1))
        perm_truth = tf.reshape(outer_boxes, (outer_size, 1, 4))
        pred_mask = tf.reshape(tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),[outer_size * FLAGS.rnn_len])
        pred_logit_r = tf.reshape(pred_logits,[outer_size * FLAGS.rnn_len, FLAGS.num_classes])
        confidences_loss = (tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logit_r, true_classes))) / outer_size*head_weights[0]
        residual = tf.reshape(perm_truth - pred_boxes * pred_mask,[outer_size, FLAGS.rnn_len, 4])
        boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size*head_weights[1]
        loss = confidences_loss + boxes_loss
    return pred_boxes,pred_confidences,loss,confidences_loss,boxes_loss,x

def training(q,save_dir):
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    gpu_options=tf.GPUOptions()
    config=tf.ConfigProto(gpu_options=gpu_options)
    learning_rate=tf.placeholder(tf.float32)
    opt=tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.9, epsilon=0.00001)
    loss, accuracy, confidences_loss, boxes_loss={},{},{},{}
    for phase in ['train', 'test']:
        # generate predictions and losses from forward pass
        x, confidences, boxes = q[phase].dequeue_many(FLAGS.batch_size)
        flags = tf.argmax(confidences, 3)

        grid_size = FLAGS.grid_width * FLAGS.grid_height
        (pred_boxes,pred_confidences,loss[phase],
            confidences_loss[phase],boxes_loss[phase],x)=losses(x,phase,boxes,flags)
        pred_confidences_r = tf.reshape(pred_confidences, [FLAGS.batch_size, grid_size, FLAGS.rnn_len, FLAGS.num_classes])
        pred_boxes_r = tf.reshape(pred_boxes, [FLAGS.batch_size, grid_size, FLAGS.rnn_len, 4])

        a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(pred_confidences_r[:, :, 0, :], 2))
        accuracy[phase] = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)

            tvars = tf.trainable_variables()
            if FLAGS.clip_norm <= 0:
                grads = tf.gradients(loss['train'], tvars)
            else:
                grads, norm = tf.clip_by_global_norm(tf.gradients(loss['train'], tvars), FLAGS.clip_norm)
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        elif phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([accuracy['train'], accuracy['test'],
                                          confidences_loss['train'], boxes_loss['train'],
                                          confidences_loss['test'], boxes_loss['test'],
                                          ])
            for p in ['train', 'test']:
                tf.scalar_summary('%s/accuracy' % p, accuracy[p])
                tf.scalar_summary('%s/accuracy/smooth' % p, moving_avg.average(accuracy[p]))
                tf.scalar_summary("%s/confidences_loss" % p, confidences_loss[p])
                tf.scalar_summary("%s/confidences_loss/smooth" % p,
                    moving_avg.average(confidences_loss[p]))
                tf.scalar_summary("%s/regression_loss" % p, boxes_loss[p])
                tf.scalar_summary("%s/regression_loss/smooth" % p,
                    moving_avg.average(boxes_loss[p]))

        if phase == 'test':
            test_image = x
            # show ground truth to verify labels are correct
            test_true_confidences = confidences[0, :, :, :]
            test_true_boxes = boxes[0, :, :, :]

            # show predictions to visualize training progress
            test_pred_confidences = pred_confidences_r[0, :, :, :]
            test_pred_boxes = pred_boxes_r[0, :, :, :]

            def log_image(np_img, np_confidences, np_boxes, np_global_step, pred_or_true):

                merged = train_utils.add_rectangles(np_img, np_confidences, np_boxes,
                                                    use_stitching=False,
                                                    rnn_len=FLAGS.rnn_len)[0]

                num_images = 10
                img_path = os.path.join(save_dir, '/%s_%s.jpg' % ((np_global_step / FLAGS.display_iter) % num_images, pred_or_true))
                print(img_path)
                misc.imsave(img_path, merged)
                return merged

            pred_log_img = tf.py_func(log_image,
                                      [test_image, test_pred_confidences, test_pred_boxes, global_step, 'pred'],
                                      [tf.float32])
            true_log_img = tf.py_func(log_image,
                                      [test_image, test_true_confidences, test_true_boxes, global_step, 'true'],
                                      [tf.float32])
            tf.image_summary(phase + '/pred_boxes', tf.pack(pred_log_img),max_images=10)
            tf.image_summary(phase + '/true_boxes', tf.pack(true_log_img),max_images=10)

    summary_op = tf.merge_all_summaries()


    return (config,loss, accuracy,summary_op, train_op,smooth_op,global_step,learning_rate,x)

def train(save_dir):

  if not os.path.exists(save_dir): os.makedirs(save_dir)
  ckpt_file = save_dir + '/save.ckpt'
  x_in = tf.placeholder(tf.float32)
  confs_in = tf.placeholder(tf.float32)
  boxes_in = tf.placeholder(tf.float32)
  q={}
  enqueue_op={}
  data_file={}
  for phase in ['train','test']:
      dtypes = [tf.float32, tf.float32, tf.float32]
      grid_size = FLAGS.grid_width*FLAGS.grid_height
      shapes = (
                [FLAGS.image_height, FLAGS.image_width, 3],
                [grid_size, FLAGS.rnn_len, FLAGS.num_classes],
                [grid_size, FLAGS.rnn_len, 4],
                )
      q[phase] = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
      enqueue_op[phase] = q[phase].enqueue((x_in, confs_in, boxes_in))
  def make_feed(d):
      return {x_in: d['image'], confs_in: d['confs'], boxes_in: d['boxes'],learning_rate:FLAGS.learning_rate}

  def thread_loop(sess, enqueue_op, phase, gen):
      for d in gen:
          sess.run(enqueue_op[phase], feed_dict=make_feed(d))

  (config,loss,accuracy,summary_op,train_op,smooth_op,global_step,learning_rate,x)=training(q,save_dir)

  saver=tf.train.Saver(max_to_keep=None)
  writer=tf.train.SummaryWriter(logdir=save_dir, flush_secs=10)
  data_file['train']=FLAGS.train_dir
  data_file['test']=FLAGS.test_dir
  with tf.Session(config=config) as sess:
      tf.train.start_queue_runners(sess=sess)
      for phase in ['train','test']:
              data_gen=train_utils.load_data_gen(data_file[phase])
              data=data_gen.next()
              sess.run(enqueue_op[phase], feed_dict=make_feed(data))
              t = tf.train.threading.Thread(target=thread_loop,
                                 args=(sess, enqueue_op, phase, data_gen))
              t.daemon = True
              t.start()

      tf.set_random_seed(1)
      sess.run(tf.initialize_all_variables())
      writer.add_graph(sess.graph)
      start=time.time()
      max_iter=1000000

      for i in xrange(max_iter):
            adjusted_lr=(FLAGS.learning_rate*0.5**
                        max(0,(i/FLAGS.learning_rate_step)-2))
            lr_feed={learning_rate:adjusted_lr}
            if i % FLAGS.display_iter != 0:
                # train network
                batch_loss_train, _ = sess.run([loss['train'], train_op],feed_dict=lr_feed)
            else:
                # test network every N iterations; log additional info
                if i > 0:
                    dt = (time.time() - start) / (FLAGS.batch_size * FLAGS.display_iter)
                start = time.time()
                (train_loss, test_accuracy, summary_str,
                    _, _) = sess.run([loss['train'], accuracy['test'],
                                      summary_op,train_op, smooth_op,
                                     ],feed_dict=lr_feed)
                writer.add_summary(summary_str,global_step=global_step.eval())
                print_str = string.join([
                    'Step: %d',
                    'lr: %f',
                    'Train Loss: %.2f',
                    'Softmax Test Accuracy: %.1f%%',
                    'Time/image (ms): %.1f'
                ], ', ')
                print(print_str %
                      (i, adjusted_lr, train_loss,
                       test_accuracy * 100, dt * 1000 if i > 0 else 0))

            if global_step.eval() % FLAGS.save_iter == 0 or global_step.eval() == max_iter - 1:
                saver.save(sess, ckpt_file, global_step=global_step)


def main():
    save_dir=FLAGS.save_dir+'/output_%s'%(datetime.datetime.now().strftime('%Y_%m_%d_%H.%M'))
    train(save_dir)

if __name__=='__main__':
    main()

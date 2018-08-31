from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data
import models


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', dest='experiment_name', default='ck_cv1_shortcut0_dz_gencfc', help='experiment_name')
parser.add_argument('--test_int', dest='test_int', type=float, default=1.0, help='test_int')
parser.add_argument('--test_int_min', dest='test_int_min', type=float, default=-1.0, help='test_int_min')
parser.add_argument('--test_int_max', dest='test_int_max', type=float, default=1.0, help='test_int_max')
parser.add_argument('--n_slide', dest='n_slide', type=int, default=10, help='n_slide')
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)
    
# model
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dz_dim  = args['dz_dim']
z_dim = args['z_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']
dz_layers  = args['dz_layers']
# testing
thres_int = args['thres_int']
test_int = args_.test_int
test_int_min = args_.test_int_min
test_int_max = args_.test_int_max
n_slide = args_.n_slide
batch_size_test = 1
# others
use_cropped_img = args['use_cropped_img']
experiment_name = args_.experiment_name



# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = tl.session()
te_data = data.ImgDataPair('./data/CK+/cross_validation1/test_peak', img_size, batch_size_test, 
                      pair=False,drop_remainder=False, shuffle=False, repeat=1, sess=sess, crop=use_cropped_img)
n_classes = len(te_data.class_to_idx)

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, z_dim=z_dim)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)

# inputs
rand_z = tf.placeholder(dtype=tf.float32, shape=[None, z_dim])
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.int32, shape=[None, ])
_b_sample = tf.one_hot(_b_sample, n_classes)
rand_iden = tf.reshape(rand_z, [-1, 2, 2, z_dim//(2**2)])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)
x_sample_rand = Gdec([rand_iden], _b_sample, is_training=False)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    raise Exception(' [*] No checkpoint!')
    
# sample
try:
    for idx, batch in enumerate(te_data):
        x = batch[0] # bsx128x128x3
        y = batch[1]
        batch_size = np.shape(x)[0]
        
        # ========== or from img path ========== #
#        img_path = './data/CK+/cross_validation1/test_peak/0/0_id12.png'
#        x = im.imread(img_path) # 128x128x3
#        x = np.expand_dims(x,0) # 1x128x128x3
#        y = np.array([0])
        # ====================================== #
        
        # prepare labels
        zero_hot = np.zeros([batch_size,n_classes])
        tmp = np.array(zero_hot, copy=True)
        tmp[range(batch_size), y] = 1
        y_hot_list = [tmp]  # the first is for reconstruction
        for i in range(n_classes):
            tmp = np.array(zero_hot, copy=True)
            tmp[:, i] = 1
            y_hot_list.append(tmp)
        
        run_inte = False
        if n_slide != 1 and batch_size == 1: # 如果做intensity, batch_size只能为1，因为每一行表示一种强度
            run_inte = True
            
        # get sample
        sample = []
        for inte in range(n_slide):
            if run_inte:
                test_int = (test_int_max - test_int_min) / (n_slide - 1) * inte + test_int_min
                
            x_list = [x, np.full((batch_size, img_size, img_size // 10, 3), -1.0)] # 输入图右边一小列黑色间隔
            rand_list = []
            for i, y_hot in enumerate(y_hot_list):
                y_hot = (y_hot * 2 - 1) * thres_int # 无论做不做intensity控制, thres_int都是必须先设置的
                
                if i > 0:
                    if run_inte:
                        y_hot[..., i - 1] = test_int
                    else:
                        y_hot[..., i - 1] = y_hot[..., i - 1] * test_int / thres_int # y_hot_list第一个标签是为了reconstruction
                
                x_sample_eval, rand_sample_eval = sess.run([x_sample,x_sample_rand], feed_dict={
                        xa_sample: x, 
                        _b_sample:y_hot,
                        rand_z: data.rand_iden([batch_size,z_dim])})
                x_list.append(x_sample_eval)
                if i>0: # 不用reconstruction
                    rand_list.append(rand_sample_eval) # 从uniform distribution生成的
            x_list += rand_list
            onebatch = np.concatenate(x_list, 2) # [batch_size_test, img_size, (img_size + (img_size//10) + img_size+...), 3)
            sample.append(onebatch)
        
        sample = np.concatenate(sample) # if n_slide==1, same to onebatch         

        save_dir = './output/%s/sample_testing' % experiment_name
        pylib.mkdir(save_dir)
        im.imwrite(im.immerge(sample, batch_size*n_slide, 1),  '%s/%d.png' % (save_dir, idx))
#        im.imwrite(sample.squeeze(0), '%s/%d.png' % (save_dir, idx)) # one row by one row
        print('%d.png done!' % (idx))
except:
    traceback.print_exc()
finally:
    sess.close()
        

    
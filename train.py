from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import json
import traceback

from functools import partial

import imlib as im
import numpy as np
import pylib
import tflib as tl
import tensorflow as tf
from scipy.io import loadmat

import models
import data
from vggface import id_preserve

# ==============================================================================
# =                                    param                                   =
# ==============================================================================
# python train.py --img_size 128 --experiment_name ck_cv1


parser = argparse.ArgumentParser()
# model
parser.add_argument('--img_size', dest='img_size', type=int, default=128)
parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=0) # skip connection, don't use when randb_ enabled 
parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
parser.add_argument('--dz_dim', dest='dz_dim', type=int, default=64)
parser.add_argument('--z_dim', dest='z_dim', type=int, default=64) # should: z_dim%4==0
parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=6)
parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=6)
parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
parser.add_argument('--dz_layers', dest='dz_layers', type=int, default=3)
# training
parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
parser.add_argument('--n_sample', dest='n_sample', type=int, default=16, help='# of sample images')
# others
parser.add_argument('--use_cropped_img', dest='use_cropped_img', action='store_true')
parser.add_argument('--experiment_name', dest='experiment_name', default='ck_cv1_shortcut0_dz_gencfc')#datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

args = parser.parse_args()
# model
img_size = args.img_size
shortcut_layers = args.shortcut_layers
inject_layers = args.inject_layers
enc_dim = args.enc_dim
dec_dim = args.dec_dim
dis_dim = args.dis_dim
dz_dim  = args.dz_dim
z_dim = args.z_dim
dis_fc_dim = args.dis_fc_dim
enc_layers = args.enc_layers
dec_layers = args.dec_layers
dis_layers = args.dis_layers
dz_layers  = args.dz_layers
# training
mode = args.mode
epoch = args.epoch
batch_size = args.batch_size
batch_size_test = 5
lr_base = args.lr
n_d = args.n_d
b_distribution = args.b_distribution
thres_int = args.thres_int
test_int = args.test_int
n_sample = args.n_sample
# others
use_cropped_img = args.use_cropped_img
experiment_name = args.experiment_name

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))


# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = tl.session()
tr_data = data.ImgDataPair('./data/CK+/cross_validation1/train', img_size, batch_size, 
                      pair=True, sess=sess, crop=use_cropped_img)
sa_data = data.ImgDataPair('./data/CK+/cross_validation1/train', img_size, n_sample, 
                               pair=False, sess=sess, crop=use_cropped_img) # for sample
te_data = data.ImgDataPair('./data/CK+/cross_validation1/test_peak', img_size, batch_size_test, 
                      pair=False,drop_remainder=False, shuffle=False, repeat=1, sess=sess, crop=use_cropped_img)
n_classes = len(tr_data.class_to_idx)

vgg_path = './data/vgg-face.mat' # download from http://www.vlfeat.org/matconvnet/pretrained/
vgg_weights = loadmat(vgg_path)

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, z_dim=z_dim)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)
Dimg = partial(models.Dimg, n_classes=n_classes, dim=dis_dim, fc_dim=dis_fc_dim, n_layers=dis_layers)
Dz = partial(models.Dz, dim=dz_dim, n_layers=dz_layers)

# inputs
lr = tf.placeholder(dtype=tf.float32, shape=[])

xa = tr_data.batch_op[0]
xap = tr_data.batch_op[1]
a = tr_data.batch_op[2]
a = tf.one_hot(a, n_classes)
b = tf.random_shuffle(a)

_a = (tf.to_float(a) * 2 - 1) * thres_int
if b_distribution == 'none':
    _b = (tf.to_float(b) * 2 - 1) * thres_int
elif b_distribution == 'uniform':
    _b = (tf.to_float(b) * 2 - 1) * tf.random_uniform(tf.shape(b)) * (2 * thres_int)
elif b_distribution == 'truncated_normal':
    _b = (tf.to_float(b) * 2 - 1) * (tf.truncated_normal(tf.shape(b)) + 2) / 4.0 * (2 * thres_int)

rand_z = tf.placeholder(dtype=tf.float32, shape=[None, z_dim])
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.int32, shape=[None, ])
_b_sample = tf.one_hot(_b_sample, n_classes)

# generate
z = Genc(xa)
xb_ = Gdec(z, _b)

xa_iden = z[-1]
rand_iden = tf.reshape(rand_z, [-1, tl.shape(xa_iden)[1], tl.shape(xa_iden)[2], z_dim//(tl.shape(xa_iden)[1]**2)])
randb_ = Gdec([rand_iden], _b)

with tf.control_dependencies([xb_, randb_]):
    xa_ = Gdec(z, _a)

# discriminate
xa_logit_gan, xa_logit_cls = Dimg(xa)
xb__logit_gan, xb__logit_cls = Dimg(xb_)

xa_iden_logit_gan = Dz(xa_iden)
rand_iden_logit_gan = Dz(rand_iden)

# discriminator losses
# for Dz
xa_iden_gan_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(xa_iden_logit_gan), 
                logits=xa_iden_logit_gan
                )
        )
rand_iden_gan_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(rand_iden_logit_gan), 
                logits=rand_iden_logit_gan
                )
        )
dz_loss = (xa_iden_gan_loss + rand_iden_gan_loss)*1.0

# for Dimg
if mode == 'wgan':  # wgan-gp
    wd = tf.reduce_mean(xa_logit_gan) - tf.reduce_mean(xb__logit_gan)
    dimg_loss_gan = -wd
    gp = models.gradient_penalty(Dimg, xa, xb_)
elif mode == 'lsgan':  # lsgan-gp
    xa_gan_loss = tf.losses.mean_squared_error(tf.ones_like(xa_logit_gan), xa_logit_gan)
    xb__gan_loss = tf.losses.mean_squared_error(tf.zeros_like(xb__logit_gan), xb__logit_gan)
    dimg_loss_gan = xa_gan_loss + xb__gan_loss
    gp = models.gradient_penalty(Dimg, xa)
elif mode == 'dcgan':  # dcgan-gp
    xa_gan_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(xa_logit_gan), 
                    logits=xa_logit_gan
                    )
            )
    xb__gan_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(xb__logit_gan), 
                    logits=xb__logit_gan
                    )
            )
    dimg_loss_gan = xa_gan_loss + xb__gan_loss
    gp = models.gradient_penalty(Dimg, xa)

xa_loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=a, logits=xa_logit_cls))
dimg_loss = dimg_loss_gan + gp * 10.0 + xa_loss_cls

# generator losses
if mode == 'wgan':
    xb__loss_gan = -tf.reduce_mean(xb__logit_gan)
elif mode == 'lsgan':
    xb__loss_gan = tf.losses.mean_squared_error(tf.ones_like(xb__logit_gan), xb__logit_gan)
elif mode == 'dcgan':
    xb__loss_gan = tf.losses.sigmoid_cross_entropy(tf.ones_like(xb__logit_gan), xb__logit_gan)

xa_iden_loss_gan = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(xa_iden_logit_gan), 
                logits=xa_iden_logit_gan
                )
        )
xb__loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=b, logits=xb__logit_cls))
xa__loss_rec = tf.losses.absolute_difference(xa, xa_)
xb__loss_idp = id_preserve(vgg_weights, xa, xb_) # identity preserve loss
g_loss = xb__loss_gan + xb__loss_cls * 10.0 + xa__loss_rec * 100.0 + xa_iden_loss_gan + xb__loss_idp

# optim
dz_var = tf.trainable_variables('Dz')
dz_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(dz_loss, var_list=dz_var)

dimg_var = tl.trainable_variables('Dimg')
dimg_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(dimg_loss, var_list=dimg_var)

g_var = tl.trainable_variables('G')
g_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)

# summary
d_summary = tl.summary({
    dz_loss: 'dz_loss',
    dimg_loss: 'dimg_loss',
    dimg_loss_gan: 'dimg_loss_gan',
    gp: 'gp',
    xa_loss_cls: 'xa_loss_cls',
}, scope='Dimg')

lr_summary = tl.summary({lr: 'lr'}, scope='Learning_Rate')

g_summary = tl.summary({
    g_loss: 'g_loss',
    xb__loss_gan: 'xb__loss_gan',
    xb__loss_cls: 'xb__loss_cls',
    xa__loss_rec: 'xa__loss_rec',
    xb__loss_idp: 'xb__loss_idp',
    xa_iden_loss_gan: 'xa_iden_loss_gan'
}, scope='G')

d_summary = tf.summary.merge([d_summary, lr_summary])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)
x_sample_rand = Gdec([rand_iden], _b_sample, is_training=False)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# iteration counter
it_cnt, update_cnt = tl.counter()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:
    # data for sampling
    xa_sample_ipt, a_sample_ipt = sa_data.get_next()
    zero_hot = np.zeros([n_sample,n_classes])
    tmp = np.array(zero_hot, copy=True)
    tmp[range(n_sample), a_sample_ipt] = 1
    b_sample_ipt_list = [tmp] # the first is for reconstruction
    for i in range(n_classes):
        tmp = np.array(zero_hot, copy=True)
        tmp[:, i] = 1
        b_sample_ipt_list.append(tmp)

    it_per_epoch = len(tr_data) // (batch_size * (n_d+1))
    max_it = epoch * it_per_epoch
    for it in range(sess.run(it_cnt), max_it):
        with pylib.Timer(is_output=False) as t:
            sess.run(update_cnt)

            # which epoch
            epoch = it // it_per_epoch
            it_in_epoch = it % it_per_epoch + 1

            # learning rate
            lr_ipt = lr_base / (10 ** (epoch // 1000)) # !!!!!
            
            n_dz = 1
            for i in range(n_dz):
                d_summary_opt, _, dz_lossv = sess.run([d_summary, dz_step, dz_loss],
                                                      feed_dict={lr: lr_ipt,
                                                                 rand_z: data.rand_iden([batch_size,z_dim])})
            # train D
            for i in range(n_d):
                d_summary_opt, _, dimg_lossv = sess.run([d_summary, dimg_step, dimg_loss],
                                                        feed_dict={lr: lr_ipt,
                                                                   rand_z: data.rand_iden([batch_size,z_dim])})
            summary_writer.add_summary(d_summary_opt, it)

            # train G
            g_summary_opt, _, g_lossv = sess.run([g_summary, g_step, g_loss], 
                                                 feed_dict={lr: lr_ipt,
                                                            rand_z: data.rand_iden([batch_size,z_dim])})
            summary_writer.add_summary(g_summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("Iter: (%d/%d) Epoch: (%d)(%d/%d) g_loss: %.4f  d_loss: %.4f dz_loss: %.4f Time: %s!" % (
                        it, max_it, epoch, it_in_epoch, it_per_epoch, g_lossv, dimg_lossv, dz_lossv, t))
            # save
            if (it + 1) % 1000 == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
                print('Model is saved at %s!' % save_path)

            # sample
            if (it + 1) % 100 == 0:
                x_sample_opt_list = [xa_sample_ipt, np.full((n_sample, img_size, img_size // 10, 3), -1.0)] # 输入图右边一小列黑色间隔
                rand_sample_opt_list = []
                for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                    _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
                    if i > 0:
                        _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
                    x_sample_eval, rand_sample_eval = sess.run([x_sample,x_sample_rand], feed_dict={
                            xa_sample: xa_sample_ipt, 
                            _b_sample:_b_sample_ipt,
                            rand_z: data.rand_iden([n_sample,z_dim])})
                    x_sample_opt_list.append(x_sample_eval)
                    if i>0: # 不用reconstruction
                        rand_sample_opt_list.append(rand_sample_eval) # 从uniform distribution生成的
                x_sample_opt_list += rand_sample_opt_list
                sample = np.concatenate(x_sample_opt_list, 2) # [n_sample, img_size, (img_size + (img_size//10) + img_size+...), 3)

                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(im.immerge(sample, n_sample, 1), '%s/Iter_(%d)_Epoch_(%d)_(%dof%d).jpg' % (save_dir, it, epoch, it_in_epoch, it_per_epoch))
except:
    traceback.print_exc()
finally:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
    print('Model is saved at %s!' % save_path)
    sess.close()


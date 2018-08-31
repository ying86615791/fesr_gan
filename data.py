# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import tflib as tl
import imlib as im
import traceback
import sys

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def rand_iden(shape,minv=-1.0,maxv=1.0):
    return np.random.uniform(
            minv,
            maxv,
            shape
            )

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def batch_dataset(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if drop_remainder: # 如果这次epoch剩下的个数不足batch_size，就把这些丢掉
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        # drop_remainder之后，每次batch的数据的shape[0]都是可知的, 即shape[0]=batch_size
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch) #设置了batch_size后，预取prefetch_batch个batch

    return dataset


def disk_image_batch_dataset(img_paths, img_paths_pair, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    """Disk image batch dataset.
    This function is suitable for jpg and png files
    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    params = (img_paths,)
    if labels is not None:
        params += (labels,)
    if img_paths_pair is not None:
        params += (img_paths_pair,)
    
    dataset = tf.data.Dataset.from_tensor_slices(params)
    
    def parse_func(path, label=None, path_pair=None):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3)
        img_pair = None
        if path_pair is not None:
            img_pair = tf.read_file(path_pair)
            img_pair = tf.image.decode_png(img_pair, 3)
        return (img, label, img_pair) # 确保parse_func返回的值个数和map_func_的参数个数一致
        
    
    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func
    
    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat)
    
    return dataset


    
class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

        self._is_eager = tf.executing_eagerly()
        self._eager_iterator = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        if self._is_eager:
            self._eager_iterator = tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _build(self, dataset, sess=None):
        self._dataset = dataset

        if self._is_eager:
            self._eager_iterator = tfe.Iterator(dataset)
        else:
            self._iterator = dataset.make_initializable_iterator()
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:
                self._sess = tl.session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op



class ImgDataPair(Dataset):
    
    def __init__(self, data_dir, img_resize, batch_size, pair, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, sess=None, crop=True):
        super(ImgDataPair, self).__init__()
        classes, self.class_to_idx = self._find_classes(data_dir)
        self.img_paths, self.labels = self.make_dataset(data_dir, self.class_to_idx, IMG_EXTENSIONS)
        if pair:
            self.img_paths_pair = self.make_pair(self.img_paths, self.labels)
        else:
            self.img_paths_pair = None
        
        offset_h = 0
        offset_w = 0
        img_size = 128
        
        def _map_func(img, label=None, img_pair=None):
            if crop:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.cast(img, tf.float32)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            
            if img_pair is not None:
                if crop:
                    img_pair = tf.image.crop_to_bounding_box(img_pair, offset_h, offset_w, img_size, img_size)
                img_pair = tf.image.resize_images(img_pair, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
                img_pair = tf.cast(img_pair, tf.float32)
                img_pair = tf.clip_by_value(img_pair, 0, 255) / 127.5 - 1
                return img, img_pair, label
            else:
                return img, label
        
        dataset = disk_image_batch_dataset(img_paths=self.img_paths,
                                           img_paths_pair=self.img_paths_pair,
                                           labels=self.labels,
                                           batch_size=batch_size,
                                           prefetch_batch=prefetch_batch,
                                           drop_remainder=drop_remainder,
                                           map_func=_map_func,
                                           num_threads=num_threads,
                                           shuffle=shuffle,
                                           buffer_size=buffer_size,
                                           repeat=repeat)
        
        self._build(dataset, sess)
        self._img_num = len(self.img_paths)
        
    def __len__(self):
        return len(self.img_paths)
        
    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def make_dataset(self, dir, class_to_idx, extensions):
        images = []
        labels = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        images.append(path)
                        labels.append(class_to_idx[target])
        return images, labels
    
    def make_pair(self, images, labels):
        labels = np.array(labels)
        idxs_dict = {} # 保存每个类别元素的下标
        for _,y in enumerate(np.unique(labels)):
            idxs_dict[y]=list(np.where(labels==y)[0])
        
        images_pair = []
        labels_pair = []
        for i in range(len(labels)):
            y = labels[i]
            _idxs_raw = idxs_dict[y]
            _idxs = _idxs_raw.copy() # 用副本，防止后面把这个list移空了
            _idxs.remove(i) # 把当前样本的下标移除, 确保不会取到同样的值
            if len(_idxs) == 0:
                _idxs = [i] # 如果这个类只有当前一个样本, 那就算了吧
                print('class %d has only one element' % y)
            # 如果用while, 在tf的多线程里会出错
            _idx_for_idxs = np.random.randint(0, len(_idxs))
            pair_idx = _idxs[_idx_for_idxs] # 从对应类别下标中随机选择一个
            if i == pair_idx:
                print('length: %d, now idx(from 0): %d, pair idx: %d, same!!!'%(len(labels), i, pair_idx))
            labels_pair.append(labels[pair_idx])
            images_pair.append(images[pair_idx])
        
        if not (labels==labels_pair).all():
            raise ValueError('labels_pair not equal to labels')
        return images_pair
            

if __name__ == '__main__':
    
    data_tr = ImgDataPair('./exp_data/CK+/cross_validation1/train', 128, 32, 
                          pair=True, repeat=1)
    data_te = ImgDataPair('./exp_data/CK+/cross_validation1/test_peak', 128, 15, 
                          pair=True,drop_remainder=False, repeat=1)
    
    images = data_tr.img_paths
    labels = data_tr.labels
    
    step = 0
    try:
        while True:
            print(step)
            xt, xpt, yt = data_tr.batch_op
            print(tl.shape(xt))
            
            x, xp, y = data_te.get_next()
            print(np.shape(x))
            step += 1
    except:
#        traceback.print_exc()
        print('Done')
    finally:
        print('This is finally')
    
        
        
        

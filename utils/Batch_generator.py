from __future__ import absolute_import
from collections import defaultdict

import numpy as np


class BatchGenerator(object):
    def __init__(self, labels, num_instances, batch_size):
        self.labels = labels
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.ids = set(self.labels)
        self.num_id = batch_size//num_instances

        self.index_dic = defaultdict(list)

        for index, cat_id in enumerate(self.labels):
            self.index_dic[cat_id].append(index)

    def __len__(self):
        return self.num_id*self.num_instances

    def batch(self):
        ret = []
        indices = np.random.choice( list(self.ids), size=self.num_id, replace=False)
        # print(indices)
        for cat_id in indices:
            t = self.index_dic[cat_id]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return ret

    def get_id(self):
        ret = self.batch()
        # print(ret)
        result = [self.labels[k] for k in ret]
        return result


def main():
    labels = np.load('/Users/wangxun/Deep_metric/labels.npy')
    num_instances = 8
    batch_size = 128
    Batch = BatchGenerator(labels, num_instances=num_instances, batch_size=batch_size)
    print(Batch.batch())

if __name__ == '__main__':
    main()
    print('Hello world')
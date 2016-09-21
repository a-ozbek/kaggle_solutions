import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar

def _row2im(row_):
    label = row_[0]
    im = (np.array(row_[1:])).reshape(28,28)
    im = im / 255.0
    return im, label

train = pd.read_csv('train.csv')


bar = progressbar.ProgressBar(max_value=train.shape[0])
im_and_label = []
for r in train.iterrows():
    n_row, row_data = r
    im_and_label.append(_row2im(row_data))
    bar.update(n_row+1)
print 'Done.'
print 'len(im_and_label):', len(im_and_label)










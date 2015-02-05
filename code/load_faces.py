""" This file is for loading face images for deep learning.
"""


try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy
import scipy.misc
import os
import cPickle as pickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def logistic_transform(A, mu, sigma):
    A[numpy.where(A == 0)] = 0.1
    Z = 1 / (1 + numpy.exp(-sigma * (A - mu)))
    Z = 2 * 255 * (Z - Z.min()) / (Z.max() - Z.min())
    Z[numpy.where(Z > 255)] = 255
    Z /= 255.
    return Z

def import_data(label, data_dir, image_dim):
    
    inputs=[]
    target=[]
    
    for (dirpath,dirnames,filenames) in os.walk(data_dir):
        print "Loading "+dirpath
        paths = [os.path.join(dirpath, f) for f in filenames if ((f[-4:] == '.png') and ('normalized' in f))]
        #image_paths += paths
        if len(paths)==0:
            continue

        # Tab indent this to do each directory separately.  This is useful when adding labels.
        # Load label file:
        try:
            print os.path.join(dirpath,'labels.csv')
            label_df = pandas.read_csv(os.path.join(dirpath,'labels.csv'))
        except:
            print "No labels file in this folder."

        # Loop through paths and store images and labels:
        for f in paths:
            print(f)
            base=os.path.basename(f)
            filename = base
            #test_target = int(os.path.basename(f).split("_")[2].split(".")[0])
            test_targets = label_df.loc[label_df['image_name']==filename]
            test_target = test_targets.iloc[0]['Smile']


            #temp = test_targets==0
            #neutral = temp.any(1)
            #print neutral.iloc[0]


            #if neutral.iloc[0]:
            if test_target>=-1:
                test_image = numpy.array(scipy.misc.imread(f))

                if (len(test_image.flatten())!=(image_dim*image_dim)):
                    continue
                #for i, row in enumerate(test_targets.iloc[0].values):
                #    print i + str(test_targets.iloc[0][i])

                #test_image2 = test_image.astype(float)
                #temp = logistic_transform(test_image2.flatten(), 140, 0.05)
                #plt.imshow(temp.reshape(image_dim,image_dim), cmap = cm.Greys_r)
                #plt.show()
                #plt.show(block=False)

                if test_target > 50:
                        test_image2 = test_image.astype(float)
                        temp = logistic_transform(test_image2.flatten(), 140, 0.05)
                        if numpy.isnan(temp).any():
                            print "NaN found :("
                            continue
                        else:
                            inputs = numpy.append(inputs, temp, axis=0)
                            target = numpy.append(target, [1], axis=0)
                elif test_target == 0:
                        test_image2 = test_image.astype(float)
                        temp = logistic_transform(test_image2.flatten(), 140, 0.05)
                        if numpy.isnan(temp).any():
                            print "NaN found :("
                            continue
                        else:
                            inputs = numpy.append(inputs, temp, axis=0)
                            target = numpy.append(target, [0], axis=0)
                elif test_target == -1:
                        test_image2 = test_image.astype(float)
                        temp = logistic_transform(test_image2.flatten(), 140, 0.05)
                        if numpy.isnan(temp).any():
                            print "NaN found :("
                            continue
                        else:
                            inputs = numpy.append(inputs, temp, axis=0)
                            target = numpy.append(target, [-1], axis=0)

    inputs = inputs.reshape(-1,(image_dim*image_dim))
    perm = numpy.random.permutation(len(inputs[:,1]))
    inputs = inputs[perm,:]
    target = target[perm]

    ## Create label mask:
    data_mask = [target >= 0]

    data_set = (inputs, target, data_mask)
    print label + " Data:"
    print "Images:      "+str(len(inputs[:,1]))
    print "Features:    "+str(len(inputs[1,:]))
    print "Labels:      "+str(len(target))
    print "  Positive Labels:       "+str(sum(target[data_mask]))
    print "  Negative Labels:       "+str(len(target) - sum(target[data_mask]))
    print "  Unlabeled Examples:    "+str(len(target) - numpy.sum(data_mask))
    
    return data_set

def load_faces(dataset, image_dim):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    
    data_dir = os.path.join(dataset)

    pickle_dir = os.path.join(dataset,'imported_data_bin.p')
    if (os.path.isfile(pickle_dir)):
        rval, test_set = pickle.load(open(pickle_dir,"rb"))
        return rval, test_set
    else:
        print "No imported data found. Loading data from images now."

        data_dir_test = os.path.join(dataset,'train')
        train_set = import_data('Training', data_dir_test, image_dim)
        pre_train_set = (train_set[0], train_set[1])
        train_set = (train_set[0][train_set[2]], train_set[1][train_set[2]])
    
        data_dir_test = os.path.join(dataset,'valid')
        valid_set = import_data('Validation', data_dir_test, image_dim)
        valid_set = (valid_set[0][valid_set[2]], valid_set[1][valid_set[2]])

        data_dir_test = os.path.join(dataset,'test')
        test_set = import_data('Testing', data_dir_test, image_dim)
        test_set = (test_set[0][test_set[2]], test_set[1][test_set[2]])

        # Make each image a row of the matrix.
        # Make the targets a column vector.

        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')        

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
        pre_train_set_x, pre_train_set_y = shared_dataset(pre_train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y), (pre_train_set_x, pre_train_set_y)]

        save_dir = os.path.join(dataset,'imported_data_bin.p')
        pickle.dump( [rval, train_set], open(save_dir,"wb"))
        return rval, test_set

if __name__ == '__main__':
    test_DBN()

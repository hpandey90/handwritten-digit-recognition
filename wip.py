import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
#http://theses.ubn.ru.nl/bitstream/handle/123456789/2620/Klep,%20D._BSc_thesis_2016.pdf?sequence=1
# settings
lr = 1e-4
total_iterations = 10000           
cross_validation_size = 1000
dropout = 0.50
batch_size = 50
data = pd.read_csv('train_rotate.csv')
raw_img_input = data.iloc[:,1:].values
raw_img_input = raw_img_input.astype(np.float)
raw_img_input = np.multiply(raw_img_input, 1.0 / 255.0)
image_size = raw_img_input.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
    
def hot_vector(ld, nc):
    num_of_labels = ld.shape[0]
    index_offset = np.arange(num_of_labels) * nc
    labels_hot_vector = np.zeros((num_of_labels, nc))
    labels_hot_vector.flat[index_offset + ld.ravel()] = 1
    return labels_hot_vector

labels_val = data[[0]].values.ravel()
count_of_labels = np.unique(labels_val).shape[0]
labels = hot_vector(labels_val, count_of_labels)
labels = labels.astype(np.uint8)
cross_validation_images = raw_img_input[:cross_validation_size]
cross_validation_labels = labels[:cross_validation_size]
training_images = raw_img_input[cross_validation_size:]
training_labels = labels[cross_validation_size:]
epochs_completed = 0
index_in_epoch = 0
num_examples = training_images.shape[0]

def param_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_feature(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def pool_max(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def next_batch(size):
    
    global training_images
    global training_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += size
    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        training_images = training_images[perm]
        training_labels = training_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = size
        assert size <= num_examples
    end = index_in_epoch
    return training_images[start:end], training_labels[start:end]

x = tf.placeholder('float', shape=[None, image_size])
y_ = tf.placeholder('float', shape=[None, count_of_labels])

# first convolutional layer
weight_c1 = param_weight([5, 5, 1, 32])
bias_c1 = bias_feature([32])

image = tf.reshape(x, [-1,image_width , image_height,1])
#noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=30, dtype=tf.float32)
#image = image+noise
h_c1 = tf.nn.relu(conv2d(image, weight_c1) + bias_c1)
h_pool1 = pool_max(h_c1)
conv_layer1 = tf.reshape(h_c1, (-1, image_height, image_width, 4 ,8))  
conv_layer1 = tf.transpose(conv_layer1, (0, 3, 1, 4,2))
conv_layer1 = tf.reshape(conv_layer1, (-1, image_height*4, image_width*8)) 

# second convolutional layer
weight_c2 = param_weight([5, 5, 32, 64])
bias_c2 = bias_feature([64])
h_c1 = tf.nn.relu(conv2d(h_pool1, weight_c2) + bias_c2)
h_pool2 = pool_max(h_c1)

# display 64 fetures in 4 by 16 grid
conv_layer2 = tf.reshape(h_c1, (-1, 14, 14, 4 ,16))  
conv_layer2 = tf.transpose(conv_layer2, (0, 3, 1, 4,2))
conv_layer2 = tf.reshape(conv_layer2, (-1, 14*4, 14*16)) 

# densely connected layer
weight_fc1 = param_weight([7 * 7 * 64, 1024])
bias_fc2 = bias_feature([1024])
h_pool2_fc = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_fc, weight_fc1) + bias_fc2)

# dropout
keep_prob_dropout = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_dropout)

# readout layer for deep net
weight_fc2 = param_weight([1024, count_of_labels])
bias_fc2 = bias_feature([count_of_labels])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, weight_fc2) + bias_fc2)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# optimisation function
training_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# evaluation
valid_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy_of_model = tf.reduce_mean(tf.cast(valid_pred, 'float'))

# prediction function
prediction_function = tf.argmax(y,1)

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
flag=input('Do you want to load pretrained model or run a new model?(yes/no)');
if flag=='yes' or flag=='YES':
    load_file=input("Enter the filename of pretrained model");
    try:
        saver = tf.train.import_meta_graph(load_file+'.meta');
        saver.restore(sess, load_file)
    except :
        print('File Does not exist')
else:
    sess.run(init)
    SAVING_FILE=input("Enter the filename to save the trained model");
# visualisation variables
training_accuracies = []
accuracy_validation = []
x_range = []
print_steps=1
for i in range(total_iterations):
    x_data, y_data = next_batch(batch_size)        
    if i%print_steps == 0 or (i+1) == total_iterations:
        train_accuracy = accuracy_of_model.eval(feed_dict={x:x_data,y_: y_data,keep_prob_dropout: 1.0})       
        if(cross_validation_size):
            validation_accuracy = accuracy_of_model.eval(feed_dict={x: cross_validation_images[0:batch_size], 
                                                                    y_: cross_validation_labels[0:batch_size],keep_prob_dropout: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            accuracy_validation.append(validation_accuracy)
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        training_accuracies.append(train_accuracy)
        x_range.append(i)
        if i%(print_steps*10) == 0 and i:
            print_steps *= 10
    sess.run(training_step, feed_dict={x: x_data, y_: y_data, keep_prob_dropout: dropout})
    
if flag=='Yes' or flag=='yes':
    check_1=input("Do you want to overwrite the file?(yes/no)");
    if(check_1=='no'):
        temp=input("File name");
        saver.save(sess,temp);
    else:
        saver.save(sess,load_file);
else:
    saver.save(sess, SAVING_FILE);
if(cross_validation_size):
    validation_accuracy = accuracy_of_model.eval(feed_dict={x: cross_validation_images,y_: cross_validation_labels,keep_prob_dropout: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, training_accuracies,'-b', label='Training')
    plt.plot(x_range, accuracy_validation,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy_of_model')
    plt.xlabel('step')
    plt.show()
test_data = pd.read_csv('test.csv').values
test_data = test_data.astype(np.float)
test_data = np.multiply(test_data, 1.0 / 255.0)

predicted_digits = np.zeros(test_data.shape[0])
for i in range(0,test_data.shape[0]//batch_size):
    predicted_digits[i*batch_size : (i+1)*batch_size] = prediction_function.eval(feed_dict={x: test_data[i*batch_size : (i+1)*batch_size], 
                                                                                keep_prob_dropout: 1.0})
np.savetxt('predicted_output.csv', 
           np.c_[range(1,len(test_data)+1),predicted_digits], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

sess.close()
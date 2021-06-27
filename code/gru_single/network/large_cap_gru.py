import tensorflow as tf
from tensorflow.contrib import rnn
import path
import pandas as pd
import datetime
import random
import numpy as np
import os

# Training Parameters
learning_rate = 0.005
training_steps = 2000
batch_size = 500
display_step = 100

# Network Parameters
num_input = 1 # the normalized returns
timesteps = 240 # timesteps
num_hidden = 50 # hidden layer num of features
num_classes = 1 # above or below the median
dropout = 0.1
threshold = tf.constant(0.5)

# tf Graph input
tf.disable_v2_behavior()
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):
    # Current data input shape: (batch_size, timesteps, num_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a gru cell with tensorflow
    gru_cell = rnn.GRUCell(num_hidden)

    # Apply the Dropout
    gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=1.0, output_keep_prob=1.0 - dropout,
                                              state_keep_prob=1.0 - dropout)

    # Get gru cell output
    outputs, states = rnn.static_rnn(gru_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.sigmoid(logits) # for prediction, [0, 1]


# Define loss and optimizer
x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
loss_op = tf.reduce_mean(x_entropy)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
delta = tf.abs((Y - prediction))
correct_pred = tf.cast(tf.less(delta, threshold), tf.int32)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# the tool to save the results
saver = tf.train.Saver()

label = list(range(timesteps * num_input)) + ['target'] + ['ticker'] + ['date']

file_prefix = 'large'
years = 9
for i in range(years):
    file_path = path.Path(__file__).parent / 'finalized_dataset/'+str(file_prefix)+'_training_set_'+str(i)+'.csv'
    with file_path.open() as dataset_file:
        data_training_input = pd.read_csv(dataset_file, index_col=0)
    file_path = path.Path(__file__).parent / 'finalized_dataset/'+str(file_prefix)+'_test_set_'+str(i)+'.csv'
    with file_path.open() as dataset_file:
        data_test_input = pd.read_csv(dataset_file, index_col=0)
    
    data_training_input.columns = label
    data_test_input.columns = label

    training_label = data_training_input.iloc[:, timesteps * num_input]
    training_data = data_training_input.iloc[:, :timesteps * num_input]
    testing_label = data_test_input.iloc[:, timesteps * num_input]
    testing_data = data_test_input.iloc[:, :timesteps * num_input]

    # Start training
    with tf.Session() as sess:
        # print the training info
        print("-------------------------------------------------------------------------------------------------------")
        print("Training the model for Training Set " + str(i) + " from " +
              datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + "...")
        print("-------------------------------------------------------------------------------------------------------")

        # Run the initializer
        sess.run(init)

        # Restore model weights from previously saved model
        if i != 0:
            load_path = saver.restore(sess, log_path)
            print("Model restored from file: %s" % save_path)

        for step in range(training_steps):
            batch_ind = random.sample(range(len(data_training_input)), batch_size)
            batch = data_training_input.iloc[batch_ind, :]

            # query the data from the data set
            batch_x = np.array(batch.iloc[:, :timesteps * num_input])
            batch_x = batch_x.reshape((batch_size, timesteps, num_input), order = 'F')
            batch_y = np.array(batch.iloc[:, timesteps * num_input])
            batch_y = batch_y.reshape((batch_size, num_classes))

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Step " + str(step) + ",bibatch Loss = " + \
                      "{:.4f}".format(loss) + ", Training Accuracy = " + \
                      "{:.3f}".format(acc))

        testing_data = np.array(testing_data).reshape((len(testing_data), timesteps, num_input), order = 'F')
        testing_label = np.array(testing_label).reshape((len(testing_label), num_classes))
        training_data = np.array(training_data).reshape((len(training_data), timesteps, num_input), order = 'F')
        training_label = np.array(training_label).reshape((len(training_label), num_classes))
        print("Overall Training Accuracy:", sess.run(accuracy, feed_dict={X: training_data, Y: training_label}))
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: testing_data, Y: testing_label}))

        log_path = os.path.join(os.getcwd(), 'model_for_period_' + str(i))
        save_path = saver.save(sess, log_path)
        print("Model saved in file: %s" % save_path)

        pred = sess.run(prediction, feed_dict={X: testing_data, Y: testing_label})
        pred = pred.reshape((1, len(pred))).tolist()[0]
        output_data = pd.DataFrame({'y_prob': pred, 'y_true': data_test_input['target'], 'Ticker': data_test_input['ticker'],
                                    'Date': data_test_input['date'], })
        output_data.to_csv('prediction/'+str(file_prefix)+'_prediction_period_'+str(i)+'.csv')
        print('Prediction for period ' + str(i) + ' successfully saved.')
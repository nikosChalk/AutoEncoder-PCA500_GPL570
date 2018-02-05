
import tensorflow as tf
import numpy
import random
from tensorflow.examples.tutorials.mnist import input_data


class AutoEncoder:

    mnist_dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
        # One-hot=True means that the output of array will have only one 1 value, and the rest will be 0. (Only one active neuron in the output layer)

    def __init__(self, layers):
        """
        Creates an AutoEncoder with the given layers which uses the Adadelta optimizer. Samples of this encoder are assumed to be in range [0, 1]
        :param layers: A list which contains the number of neurons of each layer. This list is assumed to be symmetrical and with odd size
        """
        if(len(layers)%2 == 0): # Checking if list size is odd
            raise ValueError('Number of layers must be an odd number')

        first_half = layers[0: (int)(len(layers)/2)]
        second_half = layers[((int)(len(layers)/2)+1) : len(layers)]
        if(first_half[::-1] != second_half):    # Split the list into 2 sublists and compare them. Ignore the middle element.
            raise ValueError('List not symmetrical')

        self._writer = None
        self._layers = layers
        self._d_x = layers[0]
        self._d_y = layers[(int)(len(layers)/2)]
        self._encoder_wmatrix = []  # contains Tesnors which represent the weight arrays between the encoder's layers. w[0] = weight matrix between input layer-0 and layer-1
        self._decoder_wmatrix = []  # contains Tesnors which represent the weight arrays between the decoder's layers. w[0] = weight matrix between input layer-0 and layer-1
        self._encoder_bmatrix = []   # contains Tesnors which represent the bias vector for the encoder's layers. b[i] = bias vector for layer (i+1)
        self._decoder_bmatrix = []   # contains Tesnors which represent the bias vector for the decoder's layers. b[i] = bias vector for layer (i+1)
        self._sess = tf.Session()
        self._summary_keys = ['per_batch', 'per_epoch', 'test_summary'] #keys which are used to group summaries

        print('Initializing making of Computational Graph...')
        # Making the Computational Graph
        # Define the initialization of the weights and the biases.
        for i in range(1, (int)(len(layers)/2)+1):
            self._encoder_wmatrix.append(tf.Variable(tf.truncated_normal([layers[i], layers[i-1]], stddev=0.0499), name=('encoder_wmatrix_' + str(i-1)) ))
            self._encoder_bmatrix.append(tf.Variable(tf.truncated_normal([layers[i], 1], stddev=0.0499), name=('encoder_bmatrix_' + str(i-1)) ))
        for i in range((int)(len(layers)/2)+1, len(layers)):
            self._decoder_wmatrix.append(tf.Variable(tf.truncated_normal([layers[i], layers[i-1]], stddev=0.0499), name=('decoder_wmatrix_' + str(i-(int)(len(layers)/2)-1)) ))
            self._decoder_bmatrix.append(tf.Variable(tf.truncated_normal([layers[i], 1], stddev=0.0499), name=('decoder_bmatrix_' + str(i-(int)(len(layers)/2)-1)) ))

        # Adding summaries for the weight and biases matrices
        for i in range (0, len(self._encoder_wmatrix)):
            tf.summary.histogram(self._encoder_wmatrix[i].name, self._encoder_wmatrix[i], collections=[self._summary_keys[0]])
            tf.summary.histogram(self._encoder_bmatrix[i].name, self._encoder_bmatrix[i], collections=[self._summary_keys[0]])

            tf.summary.histogram(self._decoder_wmatrix[i].name, self._decoder_wmatrix[i], collections=[self._summary_keys[0]])
            tf.summary.histogram(self._decoder_bmatrix[i].name, self._decoder_bmatrix[i], collections=[self._summary_keys[0]])

        # Defining NN's input place holder
        self._nn_inp_holder = tf.placeholder(dtype=tf.float32, shape=[self._d_x, None], name='nn_input_data')  # [d_x, z], where z can be anything.

        # Defining NN's Output
        self._encoder_op = self._encode(self._nn_inp_holder)    # [2, z]
        self._y_hat = self._decode(self._encoder_op)            # [d_x, z]

        # Defining NN's cost function
        with tf.name_scope('cost', values=[self._nn_inp_holder, self._y_hat]):
            self._cost = tf.reduce_sum(tf.square(tf.sub(self._nn_inp_holder, self._y_hat)), axis=1)  # [d_x, 1]. Sum is over the samples
            self._cost = tf.mul(tf.constant(1/2, dtype=tf.float32), self._cost)
            self._cost = tf.reduce_mean(self._cost, axis=0) # Scalar Value. Sum is over the features
        tf.summary.scalar('batch_cost', self._cost, collections=[self._summary_keys[0]])
        tf.summary.scalar('test_cost', self._cost, collections=[self._summary_keys[2]])

        # Defining NN's optimizing algorithm
        optimizer = tf.train.AdadeltaOptimizer()
        gradients = optimizer.compute_gradients(self._cost)
        with tf.name_scope('gradients', values=[gradients]):
            self._minimize_op = optimizer.apply_gradients(gradients)
        for i in range (0, len(gradients)): # Adding metrics for the gradient
            tf.summary.histogram(('gradients_for_' + gradients[i][1].name), gradients[i][0], collections=[self._summary_keys[0]])    #Gradients for Variables

        # Defining a metric for the mean epoch cost
        self._batches_cost_holder = tf.placeholder(dtype=tf.float32, shape=[None])
        with tf.name_scope(name='mean_epoch_cost_metrics', values=[self._batches_cost_holder]):
            self._mean_epoch_cost = tf.reduce_mean(self._batches_cost_holder)
            tf.summary.scalar('mean_epoch_cost', self._mean_epoch_cost, collections=[self._summary_keys[1]])

        self._summaries_per_batch = tf.summary.merge_all(key=self._summary_keys[0])
        self._summaries_per_epoch = tf.summary.merge_all(key=self._summary_keys[1])
        self._test_summaries = tf.summary.merge_all(key=self._summary_keys[2])

        print('Initialization of Computational Graph completed!\n')
        print('Initializing Variables...')
        self._sess.run(tf.global_variables_initializer())  #Initializes global variables and starts assessing the computation graph
        print('Initialization of Variables Done!\n')

    def delete(self):
        """
        Delete this Auto-encoder and release the Tensorflow resources that it has acquired.
        :return: void
        """
        self._writer.close()
        self._sess.close()
        tf.reset_default_graph()

    def _encode(self, data):
        """
        Encodes the given samples
        :param data: A Tensor of size [d_x, z], where z can be any number
        :return: A Tensor of size [d_y, z]
        """
        if (data.get_shape().dims[0] != self._d_x):
            raise ValueError('Input Tensor has wrong shape!')

        output = data
        for i in range(0, len(self._encoder_wmatrix)-1):
            output = self._fc_layer(self._encoder_wmatrix[i], output, self._encoder_bmatrix[i], op_name=('encoder_hl_' + str(i+1) + '_output'))
        output = self._enc_output_layer(self._encoder_wmatrix[len(self._encoder_wmatrix)-1], output, self._encoder_bmatrix[len(self._encoder_bmatrix)-1])
        return output

    def _decode(self, data):
        """
        Decodes the given samples
        :param data: A Tensor of size [d_y, z], where z can be any number
        :return: A Tensor of size [d_x, z]
        """
        if (data.get_shape().dims[0] != self._d_y):
            raise ValueError('Input Tensor has wrong shape!')

        output = data
        for i in range(0, len(self._decoder_wmatrix)-1):
            output = self._fc_layer(self._decoder_wmatrix[i], output, self._decoder_bmatrix[i], op_name=('decoder_hl_' + str(i+1) + '_output'))
        output = self._dec_output_layer(self._decoder_wmatrix[len(self._decoder_wmatrix)-1], output, self._decoder_bmatrix[len(self._decoder_bmatrix)-1])
        return output

    def train(self, total_epochs, batch_size):
        """
        Trains the weights and the biases of the Neural Network using the MNIST dataset
        :param total_epochs: The epochs that the NN should run. Must be >0
        :param batch_size: The size of each batch. Must be >0 and must have 0 modulo with 55000
        :return void
        """
        print('Initializing Training of NN...')
        if( (total_epochs<=0) or (batch_size <=0) or (AutoEncoder.training_samples() % batch_size != 0)):
            raise ValueError('Total epochs and batch size must be >0. batch_size must be integer multipler of ' + str(AutoEncoder.training_samples()))

        if(self._writer is None):
            self._writer = tf.summary.FileWriter('TensorBoard_logs/' + str(self._layers).replace(', ', '-') + '_epochs=' + str(total_epochs) + '_batchSize=' + str(batch_size))
            self._writer.add_graph(graph=self._sess.graph)  # Adds a visualization graph for displaying the Computation Graph

        total_batches = (int)(AutoEncoder.training_samples()/batch_size)
        print('Total Batches per epoch are: ', total_batches)

        for cur_epoch in range(0, total_epochs):
            batch_cost_list = []    # List which holds the cost (scalar value) for each batch for one epoch.
            for cur_batch in range(total_batches):
                batch_x, _ = AutoEncoder.mnist_dataset.train.next_batch(batch_size)
                batch_x = numpy.transpose(batch_x)  #result is [d_x, batch_size]
                c, _ = self._sess.run([self._cost, self._minimize_op], feed_dict={self._nn_inp_holder: batch_x})
                batch_cost_list.append(c)

                # Do not record the results of all the batches. Just a few of them.
                if((total_batches < 20) or (cur_batch % 15 == 0)):
                    cur_step = cur_epoch*total_batches + cur_batch
                    self._writer.add_summary(self._summaries_per_batch.eval(session=self._sess, feed_dict={self._nn_inp_holder: batch_x}), cur_step)

            # Defining which epochs should be recorded so that we do not have an overflow of metric data
            if( (total_epochs < 100) or (cur_epoch % 10 == 0)):
                # Define image summaries for 4 random samples. (Must be done per iteration, since this summary has a buggy implementation)
                input_img_summary = tf.summary.image('input_images_' + str(cur_epoch), tf.reshape(tf.transpose(self._nn_inp_holder), [-1, 28, 28, 1]), max_outputs=4)  # Get 4 images per epoch as a sample
                output_img_summary = tf.summary.image('output_images_' + str(cur_epoch), tf.reshape(tf.transpose(self._y_hat), [-1, 28, 28, 1]), max_outputs=4)  # Get 4 images per epoch as a sample
                img_summaries = tf.summary.merge([input_img_summary, output_img_summary])
                random_index = random.randrange(0, AutoEncoder.training_samples())
                random_inp_slice = numpy.transpose(AutoEncoder.mnist_dataset.train.images)[:, random_index:random_index+4+1]   #Taking the columns random_index up to random_index+4

                # Evaluate the img_summaries and the summaries_per_epoch. Write them afterwards.
                epoch_summ, img_summaries = self._sess.run([self._summaries_per_epoch, img_summaries],
                                                           feed_dict={self._nn_inp_holder: random_inp_slice, self._batches_cost_holder: batch_cost_list})
                self._writer.add_summary(epoch_summ, cur_epoch+1)
                self._writer.add_summary(img_summaries, cur_epoch+1)
                self._writer.flush()
            print('Current Epoch: ', (cur_epoch + 1), ' completed.')

        print('Training of NN Done!\n')
        return

    def test(self):
        """
        Uses the test dataset from the MNIST for testing. Note that train() must have been called beforehand, otherwise
        the behaviour is undefined.
        :return: The cost of the dataset
        """
        print('Initializing Testing of NN...')

        # Define image summaries for 10 random sample images, each being from a different class.
        test_images = numpy.transpose(AutoEncoder.mnist_dataset.test.images)
        test_labels = numpy.transpose(AutoEncoder.mnist_dataset.test.labels)
        sample_images = numpy.empty(shape=(self._d_x,0), dtype=float)
        sample_labels = [0] * AutoEncoder.num_of_classes()
        i = random.randint(0, AutoEncoder.test_samples()-1) #random index to start the slice.
        samples_taken = 0
        while(samples_taken < AutoEncoder.num_of_classes()):    #Take 10 different samples. Start searching from index i.
            cur_img = numpy.argmax(test_labels[:, i])   #value in range [0, 9]
            if(sample_labels[cur_img] == 0):    #Sample has not been taken.
                sample_images = numpy.concatenate((sample_images, numpy.reshape(test_images[:, i], (-1, 1))), axis=1)
                sample_labels[cur_img] = 1
                samples_taken += 1
            i = (i+1) % AutoEncoder.test_samples()

        input_img_summary = tf.summary.image('test_input_images', tf.reshape(tf.transpose(self._nn_inp_holder), [-1, 28, 28, 1]), max_outputs=10)
        output_img_summary = tf.summary.image('test_output_images', tf.reshape(tf.transpose(self._y_hat), [-1, 28, 28, 1]), max_outputs=10)
        img_summaries = tf.summary.merge([input_img_summary, output_img_summary])
        img_summaries = self._sess.run(img_summaries, feed_dict={self._nn_inp_holder: sample_images})
        self._writer.add_summary(img_summaries)

        # Calculating cost and the rest summaries
        c, test_summ = self._sess.run([self._cost, self._test_summaries], feed_dict={self._nn_inp_holder: test_images})
        self._writer.add_summary(test_summ)
        self._writer.flush()

        print('Testing of NN Done!\n')
        return c

    def _fc_layer(self, weight_matrix, layer_input, bias_matrix, op_name='fc_layer'):
        """
        Calculates the output of a fully connected layer using the relu() activation function
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param bias_matrix: A Tensor which holds the biases
        :param op_name: A name for the output of this operation
        :return: The output of this layer, which is relu((weight_matrix * input) + bias_matrix)
        """
        with tf.name_scope(op_name, values=[weight_matrix, bias_matrix, layer_input]):
            output = tf.nn.relu(tf.add(tf.matmul(weight_matrix, layer_input), bias_matrix)) # Broadcasting is used for performing the add operation
            tf.summary.histogram(op_name, output, collections=[self._summary_keys[0]])
            return output

    def _enc_output_layer(self, weight_matrix, layer_input, bias_matrix, op_name='encoder_output_layer'):
        """
        Calculates the output of the encoder, which is a fully connected layer with relu(x) activation function.
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param bias_matrix: A Tensor which holds the biases
        :param op_name: A name for the output of this operation
        :return: The output of this layer which is relu((weight_matrix * input) + bias_matrix)
        """
        return self._fc_layer(weight_matrix, layer_input, bias_matrix, op_name)

    def _dec_output_layer(self, weight_matrix, layer_input, bias_matrix, op_name='decoder_output_layer'):
        """
        Calculates the output of the decoder, which is a fully connected layer with sigmoid(x) activation function.
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param bias_matrix: A Tensor which holds the biases
        :param op_name: A name for the output of this operation
        :return: The output of this layer which is sigmoid((weight_matrix * input) + bias_matrix)
        """
        with tf.name_scope(op_name, values=[weight_matrix, bias_matrix, layer_input]):
            output = tf.nn.sigmoid(tf.add(tf.matmul(weight_matrix, layer_input), bias_matrix))  # Broadcasting is used for performing the add operation
            tf.summary.histogram(op_name, output, collections=[self._summary_keys[0]])
            return output

    @classmethod
    def training_samples(cls):
        """
        Returns the number of training samples that are being used.
        :return: The number of training samples
        """
        return 55000    #Defined by the MNIST dataset

    @classmethod
    def test_samples(cls):
        """
        Returns the number of test samples that are being used.
        :return: The number of test samples
        """
        return 10000    #Defined by the MNIST dataset

    @classmethod
    def num_of_classes(cls):
        """
        Returns the number of classes for the MNIST dataset
        :return: The number of classes
        """
        return 10    #Defined by the MNIST dataset
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
from datetime import datetime
from functions.batch_data import batch_data
from sklearn.metrics import confusion_matrix
import time
import matplotlib.pyplot as plt

from functions.read_mitbih import AamiReader, DS1DS2Reader
from functions.build_network import LSTMBuilder
from functions.evaluate_metrics import evaluate_metrics
from functions.InputParser import ParserAami, ParserDS1DS2

class ProgramRunner:
    def __init__(self):
        pass
    def parseInput(self):
        pass
    def readData(self):
        pass
    def run(self):
        self.parseInput()
        self.readData()
        args = self.args
        print(args)
        max_time = args.max_time # 5 3 second best 10# 40 # 100
        epochs = args.epochs # 300
        batch_size = args.batch_size # 10
        num_units = args.num_units
        bidirectional = args.bidirectional
        # lstm_layers = args.lstm_layers
        n_oversampling = args.n_oversampling
        checkpoint_dir = args.checkpoint_dir
        ckpt_name = args.ckpt_name
        test_steps = args.test_steps
        

        # =============================
        input_depth = self.input_depth
        char2numY = self.char2numY
        classes = self.classes
        n_classes = self.n_classes
        num2charY = self.num2charY
        y_seq_length = self.y_seq_length
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        # ====================================
        
        n_channels = 10
        # over-sampling: SMOTE
        X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
        y_train= y_train[:,1:].flatten()

        # Placeholders
        inputs = tf.placeholder(tf.float32, [None, max_time, input_depth], name = 'inputs')
        targets = tf.placeholder(tf.int32, (None, None), 'targets')
        dec_inputs = tf.placeholder(tf.int32, (None, None), 'output')

        # logits = build_network(inputs,dec_inputs=dec_inputs)
        networkBuilder = LSTMBuilder()
        logits = networkBuilder.build_network(inputs, dec_inputs, char2numY, n_channels=n_channels, input_depth=input_depth, num_units=num_units, max_time=max_time,
                    bidirectional=bidirectional)
        # decoder_prediction = tf.argmax(logits, 2)
        # confusion = tf.confusion_matrix(labels=tf.argmax(targets, 1), predictions=tf.argmax(logits, 2), num_classes=len(char2numY) - 1)# it is wrong
        # mean_accuracy,update_mean_accuracy = tf.metrics.mean_per_class_accuracy(labels=targets, predictions=decoder_prediction, num_classes=len(char2numY) - 1)

        with tf.name_scope("optimization"):
            # Loss function
            vars = tf.trainable_variables()
            beta = 0.001
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                                if 'bias' not in v.name]) * beta
            loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
            # Optimizer
            loss = tf.reduce_mean(loss + lossL2)
            optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

        nums = []
        for cl in classes:
            ind = np.where(classes == cl)[0][0]
            nums.append(len(np.where(y_train.flatten()==ind)[0]))
        # ratio={0:nums[3],1:nums[1],2:nums[3],3:nums[3]} # the best with 11000 for N
        ratio={0:n_oversampling,1:nums[1],2:n_oversampling,3:n_oversampling}
        sm = SMOTE(random_state=12,ratio=ratio)
        X_train, y_train = sm.fit_sample(X_train, y_train)

        X_train = X_train[:int(X_train.shape[0]/max_time)*max_time,:]
        y_train = y_train[:int(X_train.shape[0]/max_time)*max_time]

        X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
        y_train = np.reshape(y_train,[-1,y_test.shape[1]-1,])
        y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
        y_train = np.array(y_train)

        print ('Classes in the training set: ', classes)
        for cl in classes:
            ind = np.where(classes == cl)[0][0]
            print (cl, len(np.where(y_train.flatten()==ind)[0]))
        print ("------------------y_train samples--------------------")
        for ii in range(2):
            print(''.join([num2charY[y_] for y_ in list(y_train[ii+5])]))
        print ("------------------y_test samples--------------------")
        for ii in range(2):
            print(''.join([num2charY[y_] for y_ in list(y_test[ii+5])]))

        def test_model():
            # source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
            acc_track = []
            sum_test_conf = []
            for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_test, y_test, batch_size)):

                dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
                for i in range(y_seq_length):
                    batch_logits = sess.run(logits,
                                            feed_dict={inputs: source_batch, dec_inputs: dec_input})
                    prediction = batch_logits[:, -1].argmax(axis=-1)
                    dec_input = np.hstack([dec_input, prediction[:, None]])
                # acc_track.append(np.mean(dec_input == target_batch))
                acc_track.append(dec_input[:, 1:] == target_batch[:, 1:])
                y_true= target_batch[:, 1:].flatten()
                y_pred = dec_input[:, 1:].flatten()
                sum_test_conf.append(confusion_matrix(y_true, y_pred,labels=range(len(char2numY)-1)))

            sum_test_conf= np.mean(np.array(sum_test_conf, dtype=np.float32), axis=0)

            # print('Accuracy on test set is: {:>6.4f}'.format(np.mean(acc_track)))

            # mean_p_class, accuracy_classes = sess.run([mean_accuracy, update_mean_accuracy],
            #                                           feed_dict={inputs: source_batch,
            #                                                      dec_inputs: dec_input[:, :-1],
            #                                                      targets: target_batch[:, 1:]})
            # print (mean_p_class)
            # print (accuracy_classes)
            acc_avg, acc, sensitivity, specificity, PPV = evaluate_metrics(sum_test_conf)
            print('Average Accuracy is: {:>6.4f} on test set'.format(acc_avg))
            for index_ in range(n_classes):
                print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}".format(
                    classes[index_],
                    sensitivity[index_],
                    specificity[index_],
                    PPV[index_],
                    acc[index_]))
            print("\t Average -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}".format(np.mean(sensitivity),np.mean(specificity),np.mean(PPV),np.mean(acc)))
            return acc_avg, acc, sensitivity, specificity, PPV
        loss_track = []
        def count_prameters():
            print ('# of Params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        count_prameters()

        if (os.path.exists(checkpoint_dir) == False):
            os.mkdir(checkpoint_dir)
        # train the graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            print(str(datetime.now()))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            pre_acc_avg = 0.0
            if ckpt and ckpt.model_checkpoint_path:
                # # Restore
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                # saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
                # or 'load meta graph' and restore weights
                # saver = tf.train.import_meta_graph(ckpt_name+".meta")
                # saver.restore(session,tf.train.latest_checkpoint(checkpoint_dir))
                test_model()
            else:

                for epoch_i in range(epochs):
                    start_time = time.time()
                    train_acc = []
                    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
                        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                            feed_dict = {inputs: source_batch,
                                        dec_inputs: target_batch[:, :-1],
                                        targets: target_batch[:, 1:]})
                        loss_track.append(batch_loss)
                        train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:,1:])
                    # mean_p_class,accuracy_classes = sess.run([mean_accuracy,update_mean_accuracy],
                    #                         feed_dict={inputs: source_batch,
                    #                                               dec_inputs: target_batch[:, :-1],
                    #                                               targets: target_batch[:, 1:]})

                    # accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
                    accuracy = np.mean(train_acc)
                    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                    accuracy, time.time() - start_time))

                    if epoch_i%test_steps==0:
                        acc_avg, acc, sensitivity, specificity, PPV= test_model()

                        print('loss {:.4f} after {} epochs (batch_size={})'.format(loss_track[-1], epoch_i + 1, batch_size))
                        save_path = os.path.join(checkpoint_dir, ckpt_name)
                        saver.save(sess, save_path)
                        print("Model saved in path: %s" % save_path)

                        # if np.nan_to_num(acc_avg) > pre_acc_avg:  # save the better model based on the f1 score
                        #     print('loss {:.4f} after {} epochs (batch_size={})'.format(loss_track[-1], epoch_i + 1, batch_size))
                        #     pre_acc_avg = acc_avg
                        #     save_path =os.path.join(checkpoint_dir, ckpt_name)
                        #     saver.save(sess, save_path)
                        #     print("The best model (till now) saved in path: %s" % save_path)

                plt.plot(loss_track)
                plt.show()
            print(str(datetime.now()))
            # test_model()

class AamiRunner(ProgramRunner):
    def parseInput(self):
        parser = ParserAami()
        args = parser.setting()
        self.args = args

    def readData(self):
        args = self.args
        filename = args.data_dir
        max_time = args.max_time # 5 3 second best 10# 40 # 100
        classes= args.classes

        dataReader = AamiReader(filename, max_time=max_time,classes=classes,max_nlabel=100000)
        X, Y = dataReader.dataProcess() #11000
        print ("# of sequences: ", len(X))
        input_depth = X.shape[2]
        

        classes = np.unique(Y)
        char2numY = dict(zip(classes, range(len(classes))))
        n_classes = len(classes)
        print ('Classes: ', classes)
        for cl in classes:
            ind = np.where(classes == cl)[0][0]
            print (cl, len(np.where(Y.flatten()==cl)[0]))
        # char2numX['<PAD>'] = len(char2numX)
        # num2charX = dict(zip(char2numX.values(), char2numX.keys()))
        # max_len = max([len(date) for date in x])
        #
        # x = [[char2numX['<PAD>']]*(max_len - len(date)) +[char2numX[x_] for x_ in date] for date in x]
        # print(''.join([num2charX[x_] for x_ in x[4]]))
        # x = np.array(x)

        char2numY['<GO>'] = len(char2numY)
        num2charY = dict(zip(char2numY.values(), char2numY.keys()))

        Y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in Y]
        Y = np.array(Y)

        x_seq_length = len(X[0])
        y_seq_length = len(Y[0])- 1

        # split the dataset into the training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.args = args
        self.input_depth = input_depth
        self.classes = classes
        self.n_classes = n_classes
        self.num2charY = num2charY
        self.char2numY = char2numY
        self.x_seq_length = x_seq_length
        self.y_seq_length = y_seq_length
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

class DS1DS2Runner(ProgramRunner):
    def parseInput(self):
        parser = ParserDS1DS2()
        args = parser.setting()
        self.args = args

    def readData(self):
        args = self.args
        filename = args.data_dir
        max_time = args.max_time # 5 3 second best 10# 40 # 100
        classes= args.classes

        dataReader_train = DS1DS2Reader(filename, trainset=1, max_time=max_time, classes=classes, max_nlabel=100000)
        X_train, y_train = dataReader_train.dataProcess() #11000
        dataReader_test = DS1DS2Reader(filename, trainset=0, max_time=max_time, classes=classes, max_nlabel=100000)
        X_test, y_test = dataReader_test.dataProcess() #11000

        input_depth = X_train.shape[2]
        print ("# of sequences: ", len(X_train))

        classes = np.unique(y_train)
        char2numY = dict(zip(classes, range(len(classes))))
        n_classes = len(classes)
        print ('Classes (training): ', classes)
        for cl in classes:
            ind = np.where(classes == cl)[0][0]
            print (cl, len(np.where(y_train.flatten() == cl)[0]))

        print ('Classes (test): ', classes)
        for cl in classes:
            ind = np.where(classes == cl)[0][0]
            print (cl, len(np.where(y_test.flatten() == cl)[0]))


        char2numY['<GO>'] = len(char2numY)
        num2charY = dict(zip(char2numY.values(), char2numY.keys()))
        

        y_train = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y_train]
        y_test = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y_test]
        y_test = np.asarray(y_test)
        y_train = np.array(y_train)

        x_seq_length = len(X_train[0])
        y_seq_length = len(y_train[0]) - 1

        self.args = args
        self.input_depth = input_depth
        self.classes = classes
        self.n_classes = n_classes
        self.num2charY = num2charY
        self.char2numY = char2numY
        self.x_seq_length = x_seq_length
        self.y_seq_length = y_seq_length
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
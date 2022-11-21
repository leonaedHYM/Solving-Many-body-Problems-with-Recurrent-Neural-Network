"""
change lstm cell to tf.contrib.rnn.LSTMBlockFusedCell
change training batchsize
add tensorboard'function
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import time
import pickle
import utils
from utils import *
from data_generator import *

Ns = utils.Ns

args = utils.args 

#path
dpath = utils.dpath
test_data = utils.test_data
trpath = utils.trpath
tepath = utils.tepath
statepath = utils.statepath

file = utils.file


log_dir = './Log_for_training2/N=5_2' 
savedir = './Saved_model/Model'+str(args.version)+'/Ns='+str(Ns)+'/num_hidden='+str(args.num_hidden)+'/BS=64'+'/'

if not os.path.exists(savedir):
    os.makedirs(savedir)

#make data
if not os.path.exists(dpath+file):
    print('making data')
    make_data()
    print('data finished\n')

print('Ns='+str(Ns)+\
    '\nn_trial='+str(args.n_trial)+'_time='+str(args.time)+'_step=' + str(args.datastep)+\
    '\nn_epoch='+str(args.n_epoch)+' learning_rate='+str(args.learning_rate)+\
    '\ndata from '+str(dpath)+str(file)+'\ntrain_results at '+str(trpath)+\
    '\ntest data '+str(test_data)+str(file)+'\n')

np.random.seed(0)
tf.set_random_seed(0)

#train and test data
input_train = loader(dpath+file)  
input_test = loader(test_data)
input_train = input_train[0:args.batch_size]
input_test = input_test[0:100]

#model args
args.timeholder = input_train.shape[1]    #model time
pre_lstm_batch = input_test.shape[0]
pre_lstm_time = input_test.shape[1]


#model

def build_lstm(args, inputX, cell_fn):


    input_ = tf.concat((inputX[:,:,:,0], inputX[:,:,:,1]), axis=2)   #[batch, time, 64]

    inp_1 = tf.layers.dense(input_, args.num_hidden, activation = 'tanh')         #(batch, time , num_hidden)

    lstm_cell = cell_fn(args.num_hidden,activation = args.lstm_activation, forget_bias = args.forget_bias)       #lstm cell

    out_1, out1_states = tf.nn.dynamic_rnn(lstm_cell, inp_1, dtype=tf.float32)

    output_real = tf.layers.dense(out_1, args.num_feature)       #(batch,time,32)
    output_image = tf.layers.dense(out_1, args.num_feature)

    output_realf = tf.expand_dims(output_real, -1)    #[batch,time,32,1]
    output_imagef= tf.expand_dims(output_image, -1)    #[10,300,32,1]
    out = tf.concat([output_realf, output_imagef], axis = 3)   #[batch.time,32,2]

    return  out



class lstm(object):                                      #args

    def __init__(self, args):

        self.args = args
        #self.batch_size = batch_size

        self.cell_fn = tf.nn.rnn_cell.LSTMCell 

        self.build_graph(args)

    #describe
    def build_graph(self, args):                                 #add predict time

        self.graph = tf.Graph()

        with self.graph.as_default():
            
            self.inputX = tf.placeholder(tf.float32,shape=(None, args.timeholder, args.num_feature, 2))  #(batch_size, timeholder, num_feature, 2) 
            
            with tf.variable_scope("lstm", reuse=None):
                self.out = build_lstm(self.args, self.inputX, self.cell_fn)  #output.shape = [batch, time,32,2]

            self.loss = tf.losses.mean_squared_error(self.inputX, self.out)
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step, 
                                                     args.n_trial*0.2, 0.85, staircase=True )       #change learning_rate 
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            
            #predict
            self.pre_inputX = tf.placeholder(tf.float32,shape=(pre_lstm_batch, pre_lstm_time, args.num_feature, 2)) 
            
            with tf.variable_scope("lstm", reuse=True):
                self.pre_out = build_lstm(self.args, self.pre_inputX, self.cell_fn)
            
            self.pre_loss = tf.losses.mean_squared_error(self.pre_inputX, self.pre_out)

            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables())
            
            
#training
def run():

    model = lstm(args)
 
    loss_sum = np.array([])
    t_loss_sum = np.array([])
    loss_tr = tf.summary.scalar('loss_train', model.loss)
    loss_te = tf.summary.scalar('loss_test',model.pre_loss)
    with tf.Session(graph=model.graph) as sess:
            
        #load or new    
        if os.path.exists(savedir):
            ckpt = tf.train.get_checkpoint_state(savedir)
                
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored from:' + savedir)
            else:
                print('Initializing')
                sess.run(model.initial_op)
        print('start training')
        
        summary_writer = tf.summary.FileWriter(log_dir + 'loss', sess.graph)     #recording the data into the defined file
        #start training
        for epoch in range(args.n_epoch):
           
            start = time.time()
            feedDict = {model.inputX: input_train}

            opt, _loss, pre, summary_tr = sess.run([model.train_op, model.loss, model.out, loss_tr],
                                feed_dict=feedDict)                              #pre[batch.time,32,2]
            f = open('_loss.dat', 'wb')
            pickle.dump(_loss, f)
            f.close()             
 
            summary_writer.add_summary(summary_tr,epoch)
            end = time.time()
            delta_time = end - start
            print('epoch='+str(epoch)+' loss='+str(_loss)+' need time=' +str(delta_time)+'s')
            
            loss_sum = np.append(loss_sum, _loss)
            
            #print train figure, loss, eval_
            if epoch == 0 or epoch%args.ploteach ==0:    #plot_train and test
                
                loss_sum = vis(pre[args.testi], input_train[args.testi], args.statei, loss_sum, trpath,
                          'Result on traini='+str(args.testi) + ' state='+ str(args.statei)+ \
                          ' Epoch {}.png'.format(epoch))
                g = open('pre[args.testi].dat', 'wb')
                pickle.dump(pre[args.testi], g)
                g.close() 
                h = open('input_train[args.testi].dat', 'wb')
                pickle.dump(input_train[args.testi], h) 
                h.close() 
                #test
                pre_feedDict = {model.pre_inputX: input_test}

                t_loss, t_pre, summary_te = sess.run([model.pre_loss, model.pre_out, loss_te],
                                feed_dict=pre_feedDict)                              #pre[batch.time,32,2]
                summary_writer.add_summary(summary_te,epoch)
                print('test loss='+str(t_loss))
                t_loss_sum = np.append(t_loss_sum, t_loss)
                c = open('t_loss.dat', 'wb')
                pickle.dump(t_loss, c)
                c.close()
                #vis test
                if epoch%(args.ploteach*args.ploteach)==0:
                    t_loss_sum = vis(t_pre[args.testi], input_test[args.testi], args.statei, t_loss_sum, tepath,
                                'Result on testi='+str(args.testi) + ' state='+ str(args.statei)+ \
                                ' Epoch {}.png'.format(epoch))
                    d = open('t_pre[args.testi].dat', 'wb')
                    pickle.dump(t_pre[args.testi], d)
                    d.close
                    e = open('input_test[args.testi].dat', 'wb')
                    pickle.dump(input_test[args.testi], e)
                    e.close()  
                    visstate(t_pre[args.testi],input_test[args.testi],statepath)
   
                
            #save
            checkpoint_path = os.path.join(savedir, 'model.ckpt')
            model.saver.save(sess, checkpoint_path, global_step=epoch)
            print('Model has been saved in {}'.format(savedir))
            

    return opt, loss_sum, eval_, pre   #none, nparray.float, (nparray?!) pre[batch,time,32,2]



if __name__ == '__main__':
    opt, loss, eval_, pre = run()



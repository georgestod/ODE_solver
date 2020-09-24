# authors: Maziar Raissi (original code) and Georges Tod (modified to solve ODEs)
# last update: February 2020

import tensorflow as tf
import numpy as np
import pdb

###############################################################################
############################## Helper Functions ###############################
###############################################################################

def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]]), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
    
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

###############################################################################
################################ DE_solver Class ##############################
###############################################################################

class DE_solver:   
    def __init__(self, t_b, w_b,
                       t_f, layers,
                       lb, ub, N_train, limit_act):

        # Number of collocation points
        self.N_train = N_train
        
        # Domain Boundary
        self.lb = lb
        self.ub = ub
        
        # parameter to play on the activation function
        self.limit_act = limit_act
        
        # Init for Solution
        self.sol_init(t_b, w_b,
                      t_f, layers)
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # to store the loss per epoch
        self.loss_epochs = []


        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def sol_init(self, t_b, w_b,
                       t_f, layers):

        # Initialize NN
        self.w_weights, self.w_biases = initialize_NN(layers) # attention ICI
        
        # Initial conditions
        self.t_b = t_b # time
        self.w_b = w_b # state value at time, a first derivative needs also be passed, here we assume it is 0
        
        # Collocation points
        self.t_f = t_f

        # tf placeholders for Solution
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, 1], name='t_b_tf')
        self.w_b_tf = tf.placeholder(tf.float32, shape=[None, 1], name='w_b_tf')
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1], name='t_f_tf')
        
        
        # tf graphs for Solution
        self.w_b_pred  = self.sol_net_w(self.t_b_tf)
        self.sol_f_pred = self.sol_net_f(self.t_f_tf)
        self.sol_f_b_pred = self.sol_net_f(self.t_b_tf)
        
        # loss for Solution
        self.sol_loss = tf.reduce_sum(tf.square(self.w_b_tf - self.w_b_pred)) + \
                        tf.reduce_sum(tf.square(self.sol_f_b_pred[1]+1)) + \
                        1/self.N_train*tf.reduce_sum(tf.square(self.sol_f_pred[0]))
        
        # Optimizer for Solution
        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.sol_loss,
                             var_list = self.w_weights + self.w_biases,
                             method = 'L-BFGS-B',
                             options = {'maxiter': 10000,
                                          'maxfun': 10000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
    def sol_net_w(self, t):
        X = tf.concat([t],1)
        H = self.limit_act*(2.0*(X - self.lb)/(self.ub - self.lb) - 1.0)  # input data is normalized here
        w = neural_net(H, self.w_weights, self.w_biases)
        return w
    
    def sol_net_f(self, t):
        w = self.sol_net_w(t)
        w_t = tf.gradients(w, t)[0]     # first derivative of the state
        w_tt= tf.gradients(w_t, t)[0]
        f = w_tt +0.1*w_t +tf.sin(w)    # here is the differential equation

        return [f,w_t]
    
    def callback(self, loss):
        self.loss_epochs.append(loss)
        #print('Loss: %e' % (loss))

    def sol_train(self,N_iter):        
        tf_dict = {self.t_b_tf: self.t_b,
                   self.w_b_tf: self.w_b,
                   self.t_f_tf: self.t_f}
        
        self.sol_optimizer.minimize(self.sess, 
                                    feed_dict = tf_dict,         
                                    fetches = [self.sol_loss], 
                                    loss_callback = self.callback)        
        
    def sol_predict(self, t_star):
        u_star   = self.sess.run(self.w_b_pred, {self.t_b_tf: t_star})
        u_star_p = self.sess.run(self.sol_f_pred, {self.t_f_tf: t_star}) 
        
        # u_star:      contains the state
        # u_star_p[0]: contains the deviation from the DE
        # u_star_p[1]: contains the first derivative of the state
        
        weights = self.sess.run(self.w_weights)
        biases = self.sess.run(self.w_biases)
        
        
        return [u_star,u_star_p[1],u_star_p[0],self.loss_epochs,weights,biases]
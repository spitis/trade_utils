import numpy as np
import tensorflow as tf
import h5py
from trade_utils import yahoostore_utils

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3s(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv3reduce(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2reduce(x,W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

x = tf.placeholder('float', [None, 8, 8, 5], name='x-input')
y_ = tf.placeholder("float", shape=[None,5,11], name='y-input')
keep_prob = tf.placeholder("float")
keep_prob_early = 1 - ((1 - keep_prob) * .2)

xticker, xsp500 = tf.split(2, 2, x) #(None, 8, 4, 5)

W0 = weight_variable([3,3,5,16])
b0 = bias_variable([16])

h0_conv_ticker = tf.nn.relu(conv3s(xticker, W0) + b0)
h0_conv_sp500  = tf.nn.relu(conv3s(xsp500 , W0) + b0)

W1 = weight_variable([3,3,16,32])
b1 = bias_variable([32])

h1_conv_ticker = tf.nn.relu(conv3s(h0_conv_ticker, W1) + b1)
h1_conv_sp500  = tf.nn.relu(conv3s(h0_conv_sp500 , W1) + b1)

W2 = weight_variable([3,3,32,32])
b2 = bias_variable([32])

h2_conv_ticker = tf.nn.relu(conv3s(h1_conv_ticker, W2) + b2) #(None, 8, 4 ,32)
h2_conv_sp500  = tf.nn.relu(conv3s(h1_conv_sp500 , W2) + b2) #(None, 8, 4 ,32)

def doublemid84(tensor): #takes Nx8x4xM tensor, returns Nx8x6xM
    a, b, c, d = tf.split(2, 4, tensor)
    return tf.concat(2,[a, b, b, c, c, d])

def pad86(tensor): #takes Nx8x6xM tensor, returns Nx8x8xM
    return tf.pad(tensor, [[0,0],[0,0],[1,1],[0,0]])

h2_ticker = pad86(doublemid84(h2_conv_ticker))
h2_sp500 =  pad86(doublemid84(h2_conv_sp500 ))

#takes 2x Nx8x8xM tensors, returns 1x interleaved Nx16x8xM tensor
def interleave88(tensor1, tensor2):
    a1,a2,a3,a4,a5,a6,a7,a8 = tf.split(1, 8, tensor1)
    b1,b2,b3,b4,b5,b6,b7,b8 = tf.split(1, 8, tensor2)
    return tf.concat(1,[a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7,a8,b8])

h2_interleaved = interleave88(h2_ticker,h2_sp500)

W3 = weight_variable([2,2,32,64])
b3 = bias_variable([64])
h3_conv = tf.nn.relu(conv2reduce(h2_interleaved, W3) + b3)

W4 = weight_variable([3,3,64,64])
b4 = bias_variable([64])

h4_conv = tf.nn.relu(conv3s(h3_conv, W4) + b4) #(None, 8, 4 ,64)

W5 = weight_variable([3,3,64,64])
b5 = bias_variable([64])

h5_conv = tf.nn.relu(conv3s(h4_conv, W5) + b5) #(None, 8, 4 ,64)

W6 = weight_variable([3,3,64,512])
b6 = bias_variable([512])

h6_conv = tf.nn.relu(conv3reduce(h5_conv, W6) + b6) #(None, 6, 2 ,512)

W66 = weight_variable([2,2,512,512])
b66 = bias_variable([512])

h66_conv = tf.nn.relu(conv3reduce(h6_conv, W66) + b66) #(None, 5, 1 ,1024)

h66_flat = tf.reshape(h66_conv, [-1, 5*1*512])
h66_flat_drop = tf.nn.dropout(h66_flat, keep_prob)

W7 = weight_variable([5*1*512,2048])
b7 = bias_variable([2048])
h7_fc = tf.nn.relu(tf.matmul(h66_flat_drop, W7) + b7)

h7_fc_drop = tf.nn.dropout(h7_fc, keep_prob)

W77 = weight_variable([2048,1024])
b77 = bias_variable([1024])
h77_fc = tf.nn.relu(tf.matmul(h7_fc_drop, W77) + b77)
h77_fc_drop = tf.nn.dropout(h77_fc, keep_prob)

W8 = weight_variable([1024,1024])
b8 = bias_variable([1024])
h8_fc = tf.nn.relu(tf.matmul(h77_fc_drop, W8) + b8)
h8_fc_drop = tf.nn.dropout(h8_fc, keep_prob)

W6_1 = weight_variable([1024,11])
b6_1 = bias_variable([11])
W6_2 = weight_variable([1024,11])
b6_2 = bias_variable([11])
W6_4 = weight_variable([1024,11])
b6_4 = bias_variable([11])
W6_8 = weight_variable([1024,11])
b6_8 = bias_variable([11])
W6_16 = weight_variable([1024,11])
b6_16 = bias_variable([11])

y_1   = tf.nn.softmax(tf.matmul(h8_fc_drop, W6_1)  + b6_1)
y_2   = tf.nn.softmax(tf.matmul(h8_fc_drop, W6_2)  + b6_2)
y_4   = tf.nn.softmax(tf.matmul(h8_fc_drop, W6_4)  + b6_4)
y_8   = tf.nn.softmax(tf.matmul(h8_fc_drop, W6_8)  + b6_8)
y_16  = tf.nn.softmax(tf.matmul(h8_fc_drop, W6_16) + b6_16)

saver = tf.train.Saver()

def predict(X):
    with tf.Session() as sess:
        saver.restore(sess,'/media/silviu/spxdata/sd4/saves/sm4-10-933500 (copy) - 0.171212 - 0.163813.ckpt')
        feedval = {x: X, keep_prob: 1.0}
        preds = sess.run([y_1, y_2, y_4, y_8, y_16], feed_dict=feedval)
    return preds

bins = np.array([-2.50000000e-01,  -1.00000000e-01,
        -6.00000000e-02,  -3.00000000e-02,  -1.00000000e-02, 0,
         1.01010101e-02,   3.09278351e-02,   6.38297872e-02,
         1.11111111e-01,   2.56470588e-01])

def score_raw(preds):
    scores = []
    for pred_idx, pred in enumerate(preds):
        scores.append(np.sum(pred * bins,axis=1) / (2*(pred_idx+1)))
    scores = np.sum(np.array(scores), axis=0)
    return np.array(scores)

def get_spreads(tickers):
    df = yahoostore_utils.get_opens(tickers)
    spreads = ((df['Ask']-df['Bid']) / df['Open'])[1:].as_matrix()
    spreads[np.isnan(spreads)] = 5 #insert 5% spread for the NaNs
    return spreads

def pred_and_score_with_spreads(tickers, X):
    preds = predict(X)
    scores = score_raw(preds)
    spreads = get_spreads(tickers)
    return (scores - spreads, scores + spreads)

def top_tickers_with_spreads(tickers ,X):
    tops, bottoms = pred_and_score_with_spreads(tickers, X)
    top_tickers = tickers[np.argsort(tops)][::-1][:20]
    top_scores  = tops[np.argsort(tops)][::-1][:20]
    bottom_tickers = tickers[np.argsort(bottoms)][:20]
    bottom_scores  = bottoms[np.argsort(bottoms)][:20]

    return {"tops": (top_tickers, top_scores), "bottoms": (bottom_tickers,bottom_scores)}

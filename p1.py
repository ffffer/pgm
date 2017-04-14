import tensorflow as tf
import math
import numpy
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print "Start loading MNIST data "
mndata = input_data.read_data_sets('data/mnist')
train_image = mndata.train.images # 60000 * 784
test_image = mndata.test.images # 10000 * 784
test_label = mndata.test.labels
print "Finished loading MNIST data"

n_hidden = 512
n_input = 784
batch_size = 500
len_test = len(test_image)
len_train = len(train_image)
len_sample = 1000

print "Start initializing weights"
with tf.variable_scope("wake"):
    wake_weights = {
        '_h1': tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01, dtype=tf.float64)),
        '_h_out_sigma': tf.Variable(tf.random_normal([n_hidden, 2], stddev=0.01, dtype=tf.float64)),
        '_h_out_mu': tf.Variable(tf.random_normal([n_hidden, 2], stddev=0.01, dtype=tf.float64))
    }
    wake_biases = {
        '_b1': tf.Variable(tf.zeros([n_hidden], dtype=tf.float64)),
        '_b_out_sigma': tf.Variable(tf.zeros([2], dtype=tf.float64)),
        '_b_out_mu': tf.Variable(tf.zeros([2], dtype=tf.float64))
    }
with tf.variable_scope("sleep"):
    sleep_weights = {
        '_h1': tf.Variable(tf.random_normal([2, n_hidden], stddev=0.01, dtype=tf.float64)),
        '_h_out': tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01, dtype=tf.float64))
    }
    sleep_biases = {
        '_b1': tf.Variable(tf.zeros([n_hidden], dtype=tf.float64)),
        '_b_out': tf.Variable(tf.zeros([n_input], dtype=tf.float64))
    }
with tf.variable_scope("vae"):
    decoder_weights = {
        '_h1': tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01, dtype=tf.float64)),
        '_h_out_sigma': tf.Variable(tf.random_normal([n_hidden, 2], stddev=0.01, dtype=tf.float64)),
        '_h_out_mu': tf.Variable(tf.random_normal([n_hidden, 2], stddev=0.01, dtype=tf.float64))
    }
    decoder_biases = {
        '_b1': tf.Variable(tf.zeros([n_hidden], dtype=tf.float64)),
        '_b_out_sigma': tf.Variable(tf.zeros([2], dtype=tf.float64)),
        '_b_out_mu': tf.Variable(tf.zeros([2], dtype=tf.float64))
    }

    encoder_weights = {
        '_h1': tf.Variable(tf.random_normal([2, n_hidden], stddev=0.01, dtype=tf.float64)),
        '_h_out': tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01, dtype=tf.float64))
    }
    encoder_biases = {
        '_b1': tf.Variable(tf.zeros([n_hidden], dtype=tf.float64)),
        '_b_out': tf.Variable(tf.zeros([n_input], dtype=tf.float64))
    }
print "Finished initializing weights"


def qzx_ws(x):
    layer_1 = tf.add(tf.matmul(x, wake_weights['_h1']), wake_biases['_b1'])
    layer_1 = tf.nn.relu(layer_1)

    out_layer_mu = tf.matmul(layer_1, wake_weights['_h_out_mu']) + wake_biases['_b_out_mu']
    out_layer_sigma = tf.matmul(layer_1, wake_weights['_h_out_sigma']) + wake_biases['_b_out_sigma']

    return out_layer_mu, out_layer_sigma


def qzx_vae(x):
    layer_1 = tf.add(tf.matmul(x, decoder_weights['_h1']), decoder_biases['_b1'])
    layer_1 = tf.nn.relu(layer_1)

    out_layer_mu = tf.matmul(layer_1, decoder_weights['_h_out_mu']) + decoder_biases['_b_out_mu']
    out_layer_sigma = tf.matmul(layer_1, decoder_weights['_h_out_sigma']) + decoder_biases['_b_out_sigma']

    return out_layer_mu, out_layer_sigma


def pxz_ws(z):
    layer_1 = tf.add(tf.matmul(z, sleep_weights['_h1']), sleep_biases['_b1'])
    layer_1 = tf.nn.relu(layer_1)

    out_layer = tf.matmul(layer_1, sleep_weights['_h_out']) + sleep_biases['_b_out']
    prob = tf.nn.sigmoid(out_layer)

    return prob, out_layer


def pxz_vae(z):
    layer_1 = tf.add(tf.matmul(z, encoder_weights['_h1']), encoder_biases['_b1'])
    layer_1 = tf.nn.relu(layer_1)

    out_layer = tf.matmul(layer_1, encoder_weights['_h_out']) + encoder_biases['_b_out']
    prob = tf.nn.sigmoid(out_layer)

    return prob, out_layer


def sample_z(shape, mu, sigma):
    epi = tf.random_normal(shape=shape, dtype=tf.float64)
    z = mu + sigma * epi

    return z


def sample_x(x):
    result = tf.contrib.distributions.Bernoulli(dtype=tf.float64, p=x).sample()
    return result


def normal_prob(x, mu, sigma, standard=False):
    if standard:
        mu = tf.convert_to_tensor([[0.0, 0.0]], dtype=tf.float64)
        sigma = tf.convert_to_tensor([[1.0, 1.0]], dtype=tf.float64)
    x_sigma_x = tf.reduce_sum(((x-mu) ** 2.0) * (sigma ** 2.0 ** -1.0), 1)
    e = tf.exp(-0.5 * x_sigma_x)
    constant = 1.0 / (2.0 * math.pi) * (tf.reduce_prod(sigma, 1) ** -1.0)

    return tf.multiply(constant, e)


def log_normal_prob(x, mu, sigma):
    return -math.log(2*math.pi) - 1.0 * tf.log(tf.reduce_prod(sigma, 1)) - 0.5 * (tf.reduce_sum((
        ((x-mu)**2.0) * (sigma ** -2.0)), 1))


def evaluate(x, network_name):
    if network_name == 'vae':
        print "..........."
        z_mu_eval, z_sigma_eval = qzx_vae(x)
        z_gen_eval = sample_z([len_sample, batch_size, 2], z_mu_eval, z_sigma_eval)
        print ",,,,,,,,,,"
        print z_gen_eval
        z_gen_eval = tf.reshape(z_gen_eval, [len_sample * batch_size, 2])
        print "11111111111"
        print z_gen_eval
        x_gen_prob_eval, x_gen_eval = pxz_vae(z_gen_eval)
        print "222222222222"
    else:
        z_mu_eval, z_sigma_eval = qzx_ws(x)
        z_gen_eval = sample_z([len_sample, batch_size, 2], z_mu_eval, z_sigma_eval)
        z_gen_eval = tf.reshape(z_gen_eval, [len_sample * batch_size, 2])
        x_gen_prob_eval, x_gen_eval = pxz_ws(z_gen_eval)

    p1 = tf.exp(tf.negative(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_gen_eval), 1)))
    p2 = normal_prob(z_gen_eval, [], [], standard=True)
    q = normal_prob(z_gen_eval, z_mu_eval, z_sigma_eval)
    p2_over_q = tf.multiply(p2, q ** -1.0)
    p_over_q = tf.multiply(p2_over_q, p1)
    l_xi = tf.log(tf.reduce_mean(p_over_q))
    return l_xi


def plot(samples, name, dim):
    fig = plt.figure(figsize=(dim, dim))
    gs = gridspec.GridSpec(dim, dim)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    fig.savefig(name)
    plt.show()
    return fig


def plot_eval(line1, line2, name):
    x = numpy.arange(10)
    plt.plot(x, line1)
    plt.plot(x, line2)
    plt.savefig(name)
    plt.show()


def plot_scatter(samples, label, name):
    samples = numpy.matrix.transpose(samples)
    plt.scatter(samples[0], samples[1], c=label)
    plt.legend()
    plt.savefig(name)
    plt.show()


def max_min(coor):
    coor = numpy.matrix.transpose(coor)
    x_min = numpy.min(coor[0])
    x_max = numpy.max(coor[0])
    y_min = numpy.min(coor[1])
    y_max = numpy.max(coor[1])

    x = numpy.linspace(start=x_min, stop=x_max, num=20)
    y = numpy.linspace(start=y_min, stop=y_max, num=20)
    res = []
    for i in range(20):
        for j in range(20):
            res.append([x[i], y[j]])

    return res


def run():
    x = tf.placeholder(tf.float64, [None, 784])
    z = tf.placeholder(tf.float64, [None, 2])

    # train wake sleep
    z_mu_w, z_sigma_w = qzx_ws(x)
    z_gen_w = sample_z([1, 2], z_mu_w, z_sigma_w)
    x_gen_prob_w, x_gen_w = pxz_ws(z_gen_w)
    cross_entropy_w = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_gen_w), 1)
    var_list_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sleep')
    train_w = tf.train.AdamOptimizer().minimize(cross_entropy_w, var_list=var_list_w)

    # train sleep
    x_mu_prob_s, x_mu_s = pxz_ws(z)
    x_gen_s = sample_x(x_mu_prob_s)
    z_mu_s, z_sigma_s = qzx_ws(x_gen_s)
    likelihood = log_normal_prob(z, z_mu_s, z_sigma_s)
    var_list_s = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wake')
    train_s = tf.train.AdamOptimizer().minimize(-likelihood, var_list=var_list_s)

    # train vae
    z_mu_v, z_sigma_v = qzx_vae(x)
    z_gen_v = sample_z([1, 2], z_mu_v, z_sigma_v)
    x_gen_prob_v, x_gen_v = pxz_vae(z_gen_v)

    cross_entropy_v = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_gen_v), 1)
    kl = 0.5 * tf.reduce_sum((z_sigma_v)**2.0 + z_mu_v ** 2 - 1.0 - tf.log((z_sigma_v)**2), 1)
    lower_bound = tf.reduce_mean(cross_entropy_v + kl)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vae')
    train_e = tf.train.AdamOptimizer().minimize(lower_bound, var_list=var_list)

    # evaluation
    eval_vae= evaluate(x, 'vae')
    eval_ws = evaluate(x, 'ws')

    # plot
    x_vae, _ = pxz_vae(z)
    x_ws, _ = pxz_ws(z)

    # scatter
    z_mu_scatter_vae, z_sigma_scatter_vae = qzx_vae(x)
    z_scatter_vae = sample_z([len_test, 2], z_mu_scatter_vae, z_sigma_scatter_vae)
    z_mu_scatter_ws, z_sigma_scatter_ws = qzx_ws(x)
    z_scatter_ws = sample_z([len_test, 2], z_mu_scatter_ws, z_sigma_scatter_ws)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        eval_train_vae_line = []
        eval_test_vae_line = []
        eval_train_ws_line = []
        eval_test_ws_line = []

        for e in range(100):
            for b in range(110):
                batch = train_image[b * batch_size : b * batch_size + batch_size]

                sess.run(train_e, feed_dict={x: batch})

                sess.run(train_w, feed_dict={x: batch})
                sess.run(train_s, feed_dict={z: tf.random_normal([1, 2]).eval()})

                continue
                if b % 10 == 0:
                    print "e : " + str(e) + " | b : " + str(b)
                    lb = sess.run(lower_bound, feed_dict={x: batch})
                    loss_w = sess.run(cross_entropy_w, feed_dict={x: batch})
                    ml = sess.run(likelihood, feed_dict={z: tf.random_normal([1, 2]).eval()})
                    print lb
                    print tf.reduce_mean(loss_w).eval()
                    print ml
                    print "\n"
            continue
            if e % 10 == 0:
                print "start evaluation"
                eval_train_vae = 0.0
                eval_train_ws = 0.0
                eval_test_vae = 0.0
                eval_test_ws = 0.0

                for i in range(2):
                    eval_test_vae += (sess.run(eval_vae, feed_dict={x: test_image[i*batch_size:i*batch_size+batch_size]}) / float(len_test))
                    #print p1, p2, pq
                    #eval_test_ws += (sess.run(eval_ws, feed_dict={x: test_image[i*batch_size:i*batch_size+batch_size]}) / float(len_test))
                #for j in range(2):
                    #eval_train_vae += (sess.run(eval_vae, feed_dict={x: train_image[j*batch_size:j*batch_size+batch_size]}) / float(len_train))
                    #eval_train_ws += (sess.run(eval_ws, feed_dict={x: train_image[j*batch_size:j*batch_size+batch_size]}) / float(len_train))

                eval_train_vae_line.append(eval_train_vae)
                eval_train_ws_line.append(eval_train_ws)
                eval_test_vae_line.append(eval_test_vae)
                eval_test_ws_line.append(eval_test_ws)

                print str(eval_train_vae) + "\t" + str(eval_test_vae) + "\t" + str(eval_train_ws) + "\t" + str(eval_test_ws)
                print "-----------------------------------------"

        sample_vae = sess.run(x_vae, feed_dict={z: tf.random_normal([100, 2]).eval()})
        sample_ws = sess.run(x_ws, feed_dict={z: tf.random_normal([100, 2]).eval()})
        plot(sample_vae, "vae_10_10", 10)
        plot(sample_ws, "ws_10_10", 10)

        z_sample_vae = sess.run(z_scatter_vae, feed_dict={x: test_image})
        z_sample_ws = sess.run(z_scatter_ws, feed_dict={x: test_image})
        plot_scatter(z_sample_vae, test_label, "vae_z")
        plot_scatter(z_sample_ws, test_label, "ws_z")

        x_sample_vae = sess.run(x_vae, feed_dict={z: max_min(z_sample_vae)})
        x_sample_ws = sess.run(x_ws, feed_dict={z: max_min(z_sample_ws)})
        plot(x_sample_vae, "vae_20_20", 20)
        plot(x_sample_ws, "ws_20_20", 20)

        plot_eval(eval_train_ws, eval_test_ws, "wake_sleep_1000")
        plot_eval(eval_train_vae, eval_test_vae, "vae_1000")

run()


import tensorflow as tf
from tensorflow import keras
import numpy as np
L = keras.layers
K = keras.backend

from utils_package import utils
from utils_package import tqdm_utils
from utils_package import  keras_utils
from tqdm import tqdm

import zipfile
import featureExtraction
import dataPreparation

import os
import random


train_img_embeds, train_img_fns = featureExtraction.get_train_features()
val_img_embeds, val_img_fns = featureExtraction.get_val_features()

vocab = dataPreparation.generate_vocabulary()
vocab_inverse = {idx: w for w, idx in vocab.items()}

PAD = "#PAD#"
IMG_EMBED_SIZE = train_img_embeds.shape[1]
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = vocab[PAD]


batch_size = 64
n_epochs = 12
n_batches_per_epoch = 1000
n_validation_batches = 100 

MAX_LEN = 20  

np.random.seed(42)
random.seed(42)

s = keras_utils.reset_tf_session()

def generate_batch(images_embeddings, indexed_captions, batch_size, max_len=None):
    sample_index = random.sample(list(range(len(images_embeddings))),batch_size)
    batch_image_embeddings = images_embeddings[sample_index]
    batch_captions = [random.choice(caption) for caption in indexed_captions[sample_index]]
    batch_captions_matrix = dataPreparation.batch_captions_to_matrix(batch_captions,vocab[PAD],max_len)
    
    return {decoder.img_embeds: batch_image_embeddings, 
            decoder.sentences: batch_captions_matrix}

class decoder:

    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])

    sentences = tf.placeholder('int32', [None, None])
    
    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')

    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')

    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    
    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")

    token_logits = L.Dense(len(vocab),
                           input_shape=(None, LOGIT_BOTTLENECK))
    
    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))

    word_embeds = word_embed(sentences[:,:-1,]) #32*20*100
    
    hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

    
    flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS])

    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))
    
    flat_ground_truth = tf.reshape(sentences[:,1:],[-1,])

    flat_loss_mask = tf.cast(tf.not_equal(flat_ground_truth,pad_idx),tf.float32)

    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth, 
        logits=flat_token_logits
    )

    loss = tf.reduce_sum(tf.multiply(xent, flat_loss_mask)) / tf.reduce_sum(flat_loss_mask)


optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(decoder.loss)

saver = tf.train.Saver()

s.run(tf.global_variables_initializer())

train_captions, val_captions = dataPreparation.get_captions()

train_captions_indexed = dataPreparation.caption_tokens_to_indices(train_captions, vocab)
val_captions_indexed = dataPreparation.caption_tokens_to_indices(val_captions, vocab)

train_captions_indexed = np.array(train_captions_indexed)
val_captions_indexed = np.array(val_captions_indexed)


for epoch in range(n_epochs):
    
    train_loss = 0
    pbar = tqdm(range(n_batches_per_epoch))
    counter = 0
    for _ in pbar:
        train_loss += s.run([decoder.loss, train_step], 
                            generate_batch(train_img_embeds, 
                                           train_captions_indexed, 
                                           batch_size, 
                                           MAX_LEN))[0]
        counter += 1
        pbar.set_description("Training loss: %f" % (train_loss / counter))
        
    train_loss /= n_batches_per_epoch
    
    val_loss = 0
    for _ in range(n_validation_batches):
        val_loss += s.run(decoder.loss, generate_batch(val_img_embeds,
                                                       val_captions_indexed, 
                                                       batch_size, 
                                                       MAX_LEN))
    val_loss /= n_validation_batches
    
    print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

    saver.save(s, utils.get_checkpoint_path(epoch))
    
print("Finished!")


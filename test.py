import tensorflow as tf
import numpy as np 
import zipfile
import matplotlib.pyplot as plt


from train import decoder
import train
import utils
import featureExtraction
import dataPreparation

LSTM_UNITS = 300
IMG_SIZE = 299


class final_model:
    encoder, preprocess_for_model = featureExtraction.get_cnn_encoder()
    train.saver.restore(train.s, utils.get_checkpoint_path())  
    
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

    img_embeds = encoder(input_images)

    init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)
    
    current_word = tf.placeholder('int32', [1], name='current_input')

    word_embed = decoder.word_embed(current_word)

    new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))

    new_probs = tf.nn.softmax(new_logits)

    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)


def generate_caption(image, t=1, sample=False, max_len=20):
    train.s.run(final_model.init_lstm, 
          {final_model.input_images: [image]})
    
    caption = [train.vocab[dataPreparation.START]]
    
    for _ in range(max_len):
        next_word_probs = train.s.run(final_model.one_step, 
                                {final_model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()
        
        next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

        if sample:
            next_word = np.random.choice(range(len(train.vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == train.vocab[dataPreparation.END]:
            break
       
    return list(map(train.vocab_inverse.get, caption))


def apply_model_to_image_raw_bytes(raw):
    img = utils.decode_image_from_buf(raw)
    fig = plt.figure(figsize=(7, 7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    print(' '.join(generate_caption(img)[1:-1]))
    plt.show()

def show_valid_example(val_img_fns, example_idx=0):
    zf = zipfile.ZipFile("./test_data/val2014_sample.zip")
    all_files = set(val_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    apply_model_to_image_raw_bytes(zf.read(example))


_ = np.array([0.5, 0.4, 0.1])
for t in [0.01, 0.1, 1, 10, 100]:
    print(" ".join(map(str, _**(1/t) / np.sum(_**(1/t)))), "with temperature", t)    

show_valid_example(train.val_img_fns, example_idx=100)
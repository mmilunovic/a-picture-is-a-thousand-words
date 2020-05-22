import zipfile
import  json
from collections import defaultdict

import featureExtraction


# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"


# extract captions from zip
def get_captions_for_fns(fns, zip_fn, zip_json_path):
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]}
    fn_to_caps = defaultdict(list)
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))
    

def get_captions():
    _, train_img_fns = featureExtraction.get_train_features()
    _, val_img_fns = featureExtraction.get_val_features()

    train_captions = get_captions_for_fns(train_img_fns, "captions_train-val2014.zip", 
                                        "annotations/captions_train2014.json")

    val_captions = get_captions_for_fns(val_img_fns, "captions_train-val2014.zip", 
                                        "annotations/captions_val2014.json")
                                    

    return train_captions, val_captions


# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))

def generate_vocabulary():
    vocab =[]
    token_count = dict()

    train_captions, _ = get_captions()
    
    for cap in train_captions:
        for sen in cap:
            for token in split_sentence(sen):
                token_count[token] = token_count.get(token,0) + 1
                
    for token in token_count:
        if token_count.get(token) >= 5:
            vocab.append(token)
    
    vocab.extend([PAD,UNK,START,END])

    
    return {token: index for index, token in enumerate(sorted(vocab))}
    
def caption_tokens_to_indices(captions, vocab):
    res = []
    
    for caption in captions:
        tmp_caption = []
        for sen in caption:
            tmp_sen = []
            tmp_sen.append(vocab.get(START))
            for token in split_sentence(sen):
                if vocab.get(token):
                    tmp_sen.append(vocab.get(token))
                else:
                    tmp_sen.append(vocab.get(UNK))
            tmp_sen.append(vocab.get(END))
            tmp_caption.append(tmp_sen)
        res.append(tmp_caption)
    
    return res

# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    if max_len is None:
        columns = max(map(len, batch_captions))
    else:
        columns = min(max_len, max(map(len, batch_captions)))
        
    matrix = np.ones([len(batch_captions),columns])*pad_idx
    
    for i in range(len(batch_captions)):
        j = min(len(batch_captions[i]), columns)
        matrix[i, :j] = batch_captions[i][:j]
        
    return matrix
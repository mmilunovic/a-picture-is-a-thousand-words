# A picture is worth a thousand (coherent) words

Implementation of CNN-RNN architecture for image caption generation proposed in [this paper](https://arxiv.org/abs/1411.4555). 

![](https://github.com/mmilunovic/a-picture-is-a-thousand-words/blob/master/resources/description.png)

![](https://github.com/mmilunovic/a-picture-is-a-thousand-words/blob/master/resources/multiple.png)

[Google AI Blog](https://ai.googleblog.com/2014/11/a-picture-is-worth-thousand-coherent.html) about this problem.



## Getting started

```bash
git clone https://github.com/mmilunovic/a-picture-is-a-thousand-words.git
pip install -r requirements.txt
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Training the model yourself

If you want to train the model by yourself, you'll need to download training and validation datasets and place them in the train_data and test_data directories:

* [Train images](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)
* [Validation images](http://msvocds.blob.core.windows.net/coco2014/val2014.zip)
* [Captions for both train and validation](http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip)

## References

* [Show And Tell Paper](https://arxiv.org/abs/1411.4555) - Original paper
* [Advanced Machine Learning Course](https://www.coursera.org/specializations/aml) - Final project for this course


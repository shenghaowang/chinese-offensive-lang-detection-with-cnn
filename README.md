# chinese-offensive-lang-detection-with-cnn
Pytorch implementation of the CNN for Chinese offensive language detection

## Setup
Python >3.8 is required.

### Create virtual environment
```
virtualenv coldcnn_env
source coldcnn_env/bin/activate
pip install -r requirements.txt
python -m spacy download zh_core_web_md
```

### Install the `pre-commit` hooks
```
pre-commit install
pre-commit run --all-files
```

## Preprocess
To remove the stopwords from the text messages, a filter can be applied over the [CJK Unified Ideographs](https://unicode-table.com/en/blocks/cjk-unified-ideographs/). CJK stands for Chinese, Japanese and Korean. Code ranges related to CJK include the follows.

```
31C0—31EF CJK Strokes
31F0—31FF Katakana Phonetic Extensions
3200—32FF Enclosed CJK Letters and Months
3300—33FF CJK Compatibility
3400—4DBF CJK Unified Ideographs Extension A
4DC0—4DFF Yijing Hexagram Symbols
4E00—9FFF CJK Unified Ideographs
```

## Model architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 16, 108]          14,416
              ReLU-2              [-1, 16, 108]               0
         MaxPool1d-3               [-1, 16, 53]               0
            Conv1d-4              [-1, 16, 107]          19,216
              ReLU-5              [-1, 16, 107]               0
         MaxPool1d-6               [-1, 16, 52]               0
            Conv1d-7              [-1, 16, 106]          24,016
              ReLU-8              [-1, 16, 106]               0
         MaxPool1d-9               [-1, 16, 51]               0
          Flatten-10                 [-1, 2496]               0
          Dropout-11                 [-1, 2496]               0
           Linear-12                  [-1, 128]         319,616
          Dropout-13                  [-1, 128]               0
           Linear-14                    [-1, 1]             129
          Sigmoid-15                    [-1, 1]               0
================================================================
Total params: 377,393
Trainable params: 377,393
Non-trainable params: 0
----------------------------------------------------------------
```

## References
* [COLD: A Benchmark for Chinese Offensive Language Detection](https://arxiv.org/abs/2201.06025)
* [Find all Chinese text in a string using Python and Regex](https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex)

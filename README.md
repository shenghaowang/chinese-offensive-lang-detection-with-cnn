# chinese-offensive-lang-detection-with-cnn
Pytorch implementation of the CNN for Chinese offensive language detection

## Installation
Python >3.8 is required.

```
virtualenv coldcnn_env
source coldcnn_env/bin/activate
pip install -r requirements.txt
python -m spacy download zh_core_web_md
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

## References
* [COLD: A Benchmark for Chinese Offensive Language Detection](https://arxiv.org/abs/2201.06025)
* [Find all Chinese text in a string using Python and Regex](https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex)
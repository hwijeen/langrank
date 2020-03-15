import re
import string
from functools import reduce, partial

import emoji

def compose(*funcs):
    """" Compose functions so that they are applied in chain. """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs[::-1])

def _limit_pattern(sent, pattern, keep_num):
    if pattern in string.punctuation:
        re_pattern = re.escape(pattern)
    else:
        re_pattern = f'(({pattern})[\s]*)'
        pattern = pattern + ' '
    pattern_regex = re_pattern + '{' + str(keep_num+1) + ',}'
    return re.sub(pattern_regex, lambda match: pattern * keep_num, sent)

def limit_punctuations(sent, keep_num):
    puncs = ['!', '?', '.']
    for p in puncs:
        sent = _limit_pattern(sent, p, keep_num)
    return sent

def lower_hashtags(sent):
    """ e.g.  #MAGA -> #maga """
    return re.sub('#[\S]+', lambda match: match.group().lower(), sent)

def replace_urls(sent):
    return sent.replace('URL', 'http')

def remove_mentions(sent):
    sent = _limit_pattern(sent, '@USER', 0)
    sent = _limit_pattern(sent, 'USER', 0)
    sent = _limit_pattern(sent, 'US', 0)
    return sent

def remove_url(sent):
    return sent.replace(' URL', '')

def remove_retweet(sent):
    return re.sub('^RT', '', sent)

def remove_enter(sent):
    return sent.replace('<LF>', '')

def remove_hashtags(sent):
    return re.sub('#[\S]+', '', sent)

def replace_emojis(sent):
    """ e.g. smiling emoticon -> :smiley_face: """
    return emoji.demojize(sent)

def remove_emojis(sent):
    # Textify emojis first
    sent = replace_emojis(sent)
    # Replace to empty string
    return re.sub(':[\S]+:', '', sent)

def remove_punctuations(sent):
    punc = '[' + string.punctuation + '(&amp)' + ']'
    return re.sub(punc, '', sent)

def remove_numbers(sent):
    return re.sub('[0-9]+', '', sent)

def build_preprocess():
    funcs = [remove_enter,remove_numbers, remove_punctuations, remove_emojis,
             remove_hashtags, remove_retweet, remove_url, remove_mentions]
    return compose(*funcs)

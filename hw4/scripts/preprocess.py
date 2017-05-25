import re
REPLACE_CHAR = r'[-\"_:\+\-()\[\]<>\\]|(\.){2,}'
PREPEND_SPACE = r'(\!|\?|\.|\,)'

def line_preprocess(s):
    ret = re.sub(r'\' | \'|^\'| \'$', ' ', s)
    # ret = re.sub(r'(?<=(\.| ))\'', '', ret)
    ret = re.sub(REPLACE_CHAR, '', ret)
    ret = re.sub(PREPEND_SPACE, r' \1', ret)
    ret = '<bos> ' + ret.lower() + ' <eos>'
    return ' '.join(ret.split())



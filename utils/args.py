import parser

def str2bool(v):
    return v.lower() in ('true')

def str_list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(',')]

def int_list(value):
    return [int(num) for num in value.split(',')]

def add_argument_group(parser, name):
    arg = parser.add_argument_group(name)
    return arg


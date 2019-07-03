import collections


def flatten(iterable):
    '''
    Function to replace the deprecated `compiler.ast.flatten` function. This
    one was written by Gareth Latty and posted on Stack Overflow, so credit
    goes to them.
    '''
    for el in iterable:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            yield from flatten(el)
        else:
            yield el

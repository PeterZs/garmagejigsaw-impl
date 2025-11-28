from easydict import EasyDict as edict


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(indent_cnt=0)
def print_easydict(inp_dict: edict):
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            print("{}{}:".format(" " * 2 * print_easydict.indent_cnt, key))
            print_easydict.indent_cnt += 1
            print_easydict(value)
            print_easydict.indent_cnt -= 1

        else:
            print(
                "{}{}: {}".format(
                    " " * 2 * print_easydict.indent_cnt, key, value
                )
            )



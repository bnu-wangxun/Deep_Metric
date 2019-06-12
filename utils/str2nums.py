from __future__ import absolute_import


def chars2nums(a):
    nums_list = ['0']
    for k in a:
        # print(k)
        if k == ',':
            nums_list.append('0')
        else:
            nums_list[-1] = nums_list[-1] + k

    return [int(char_) for char_ in nums_list]
    

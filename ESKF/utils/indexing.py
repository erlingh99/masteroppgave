from functools import cache


@cache
def block_3x3(i: int, j: int):
    """used to generate 3x3 block slices
    This can be usefull to fill out A and G in (10.68)

    arr[block33(0,1)] == arr[0:3, 3:6]
    arr[block33(1,2)] == arr[3:6, 6:9]
    ... 

    Args:
        i (int): row in (10.68)
        j (int): column in (10.68)

    Returns:
        [type]: [description]
    """
    return slice(i*3, (i+1)*3), slice(j*3, (j+1)*3)

import random

array = list(range(50))
clean_array = array.copy()
array.extend([50, 47])

random.shuffle(array)
random.shuffle(clean_array)

def find_duplicates(arr:array):
    """
    if the array contains a duplicate, find it
    :param arr: the array
    :return: the duplicate number, or None if no duplicates
    """
    if len(arr) < 2: return None

    tortise = arr[0]
    hare = arr[1]
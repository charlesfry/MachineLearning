import re
from functools import reduce
def main():
    nums = input()
    nums = re.split('\s+', nums)
    nums = [int(i) for i in nums]
    nums = reduce(lambda x,y: x+y, nums)
    return nums

if __name__ == '__main__':
    print(main())
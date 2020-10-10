# apparently keras wont open files with a name consisting of all numbers
# so let's change that
import os


def rename(directory) :
    for root,dirs,files in os.walk(directory) :
        for file in files :

            # see if it's still just a number
            name = file.split('.')[0]
            try :
                 _ = int(name)
            except ValueError :
                continue

            os.rename(f'{directory}/{file}',f'{directory}/img_{file}')


if __name__ == '__main__' :
    dir = './input/poo'
    rename(dir)

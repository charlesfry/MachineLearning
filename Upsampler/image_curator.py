import os
from PIL import Image

try:
    os.makedirs('./input/train_scale/')
except OSError:
    pass

test_path = './input/test_scale/'

keep_size = (640, 480)

kept_files = 0
for root,_,files in os.walk(test_path) :
    for file in files :
        path = os.path.join(root,file)
        img = Image.open(path)
        if img.size != keep_size :
            os.remove(path)
            continue
        print(os.path.splitext(path[1]))
        # now resize and save

        kept_files += 1

print(kept_files)


def main() :
    pass

if __name__ == '__main__' :
    main()
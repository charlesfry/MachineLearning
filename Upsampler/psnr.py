from warnings import warn
import numpy as np


def psnr(img1, img2):
    assert img1.shape == img2.shape, f'Images must be of the same shape. You passed images of shapes {img1.shape},{img2.shape}'
    shape = list(img1.shape)

    if len(img1.shape) != 2:
        print(f'[WARNING] input shape is of dimension {len(img1.shape)}. Did you mean to do this?')

    assert shape is not None, f'shape of image must be 2-dimensional. Shape: {img1.shape}'

    diff = img1 - img2

    prod = 1
    for i in shape:
        prod *= i

    mse = np.sum(diff ** 2) / prod

    if mse == 0:
        warn('[WARNING] error is 0, making score inf. Did you accidently score the image against itself?')
        return np.inf

    # R**2 == 255**2 == 65025

    return 10 * np.log(65025 / mse)


if __name__ == '__main__':
    img1 = np.random.randn(64, 64)
    img2 = np.random.randn(64, 64)

    print(psnr(img1, img2))
    print(psnr(img1, img1.copy()))

import PIL
import PIL.Image
import numpy as np

array = np.array(
    [
        [1.0, -1.0, -1.0, 0.0, 1.0, 0.0, -1.0, -1.0],
        [1.0, 1.0, -0.0, -1.0, -0.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -0.0, -1.0, -0.0, 1.0, 1.0],
        [-1.0, -1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0, 0.0, 1.0, 0.0, -1.0, -1.0],
        [1.0, 1.0, -0.0, -1.0, -0.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -0.0, -1.0, -0.0, 1.0, 1.0],
        [-1.0, -1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 1.0],
    ]
)

print(array, array.shape)

print(array.astype(np.uint8))
print(array.astype(np.uint8).astype(np.int8))

img = PIL.Image.fromarray(
    array.astype(np.uint8),
    mode="L",
)

img.save("test_mask.png")

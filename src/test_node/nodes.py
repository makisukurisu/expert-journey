import ast
from inspect import cleandoc
import io
import os
from pathlib import Path
from typing import Generator, Literal

import PIL.ImageFilter
from server import PromptServer

import PIL.Image
import numpy
import torch
import PIL

import torchvision.transforms as transforms

import folder_paths
import numpy as np
from numpy import ndarray

import scipy.fftpack

to_tensor = transforms.Compose([transforms.PILToTensor()])
from_tensor = transforms.Compose([transforms.ToPILImage()])


class BaseNode:
    """
    Base node class
    """

    CATEGORY = "Steganography"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        raise NotImplementedError("Implement inputs for your node")

    @classmethod
    def IS_CHANGED(cls, *args):
        pass

    RETURN_TYPES: tuple
    RETURN_NAMES: tuple

    # Shown in UI
    DESCRIPTION = cleandoc(__doc__ or "No description")

    # That will be called
    FUNCTION: str

    OUTPUT_NODE: bool = False


class ImageConverter:
    @staticmethod
    def tensor_to_PIL(
        tensor: "torch.Tensor",
        mode: Literal["RGB", "YCbCr", "L"] = "RGB",
        bits: int = 8,
    ) -> list[PIL.Image.Image]:
        max_value = (2**bits) - 1
        imported_list = []

        for _, image in enumerate(tensor):
            i = 255.0 * image.cpu().numpy()
            img = PIL.Image.fromarray(
                numpy.clip(i, 0, max_value).astype(numpy.uint8),
                mode=mode,
            )
            imported_list.append(img)

        return imported_list

    @staticmethod
    def PIL_to_tensor(
        images: list[PIL.Image.Image],
        bits: int = 8,
    ) -> "torch.Tensor":
        max_value = (2**bits) - 1

        if len(images) == 1:
            return torch.from_numpy(numpy.array(images[0]) / max_value)[None,]

        return torch.cat(
            [torch.from_numpy(numpy.array(img) / max_value) for img in images],
            dim=0,
        )


class BaseEmbedder:
    def __init__(
        self,
        use_channel: int,
        block_height: int,
        block_width: int,
    ) -> None:
        self.use_channel = use_channel
        self.block_height = block_height
        self.block_width = block_width

    @staticmethod
    def message_to_generator(message: str) -> Generator[int, None, None]:
        # yields bits MSB-first for every byte in message
        data = message.encode("utf-8")
        for byte in data:
            for i in range(7, -1, -1):
                yield (byte >> i) & 1

    @staticmethod
    def file_to_generator(
        file_path: str,
    ) -> Generator[int, None, None]:
        with open(file_path, mode="rb") as file:
            while byte := file.read(1):
                value = byte[0]
                for i in range(7, -1, -1):
                    yield (value >> i) & 1

    @staticmethod
    def block_generator(
        data: np.ndarray,
        x: int = 8,
        y: int = 8,
    ) -> Generator[np.ndarray, None, None]:
        for i in range(0, data.shape[0], y):
            for j in range(0, data.shape[1], x):
                yield data[i : i + y, j : j + x]

    def block_list_to_image(
        self,
        blocks: list[np.ndarray],
        image: np.ndarray,
    ) -> np.ndarray:
        # Reshape list of blocks into an image (similar to existing.impl)
        channel = image[self.use_channel]

        rows = []
        row = []

        for block in blocks:
            row.append(block)
            if (
                block.shape[1] < self.block_width
                or len(row) * self.block_width == channel.shape[1]
            ):
                rows.append(np.hstack(row))
                row.clear()

        new_blocks = np.vstack(rows)

        channels = []
        for n, band in enumerate(image):
            if n == self.use_channel:
                data = list(new_blocks)
            else:
                data = list(band)

            array = np.array(
                data,
                dtype=np.int16,
            ).reshape(
                [
                    channel.shape[0],
                    channel.shape[1],
                ]
            )
            channels.append(array)

        combined_colors = np.zeros(
            (channel.shape[0], channel.shape[1], image.shape[0]),
            dtype=np.int16,
        )

        for i, channel in enumerate(channels):
            combined_colors[:, :, i] = channel

        return combined_colors


class MultiEmbedder(BaseEmbedder):
    def __init__(
        self,
        use_channel: int,
        block_height: int,
        block_width: int,
        code_words: list[np.ndarray],
    ) -> None:
        super().__init__(use_channel, block_height, block_width)

        self.code_words = code_words
        self.bits_per_block = len(code_words)

    def embed_info(
        self,
        image: ndarray,
        data: Generator[int, None, None],
    ) -> ndarray:
        exhausted = False
        channel = image[self.use_channel]

        block_gen = self.block_generator(
            data=channel,
            x=self.block_width,
            y=self.block_height,
        )

        new_blocks = []

        for block in block_gen:
            if exhausted:
                new_blocks.append(block)
                continue

            if block.shape != (self.block_height, self.block_width):
                new_blocks.append(block)
                continue

            bits = []

            for _ in range(self.bits_per_block):
                try:
                    bits.append(next(data))
                except StopIteration:
                    exhausted = True
                    break

            for n, bit in enumerate(bits):
                modification = self.code_words[n].copy().astype(np.int16)

                if bit == 0:
                    modification *= -1

                block = (
                    (block.astype(np.int16) + modification)
                    .clip(0, 255)
                    .astype(np.uint8)
                )

            new_blocks.append(block)

        return self.block_list_to_image(
            new_blocks,
            image,
        )


class ToYCbCr(BaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "An image to which the information will be embedded"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("converted",)

    FUNCTION = "to_ycbcr"

    OUTPUT_NODE = True

    def to_ycbcr(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        # Multiplying by 255 and dividing is necessart for torch-numpy conversion
        imported_list = ImageConverter.tensor_to_PIL(
            tensor=images,
        )
        converted_list = []

        for image in imported_list:
            converted_list.append(image.convert(mode="YCbCr"))

        return (ImageConverter.PIL_to_tensor(images=converted_list),)


# New node: convert YCbCr -> RGB (inverse of ToYCbCr)
class ToRGB(BaseNode):
    CATEGORY = "Utils"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "YCbCr image(s) to convert back to RGB"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("converted",)
    FUNCTION = "to_rgb"
    OUTPUT_NODE = True

    def to_rgb(self, images: torch.Tensor) -> tuple[torch.Tensor]:
        imported_list = ImageConverter.tensor_to_PIL(tensor=images, mode="YCbCr")
        converted = [img.convert(mode="RGB") for img in imported_list]
        return (ImageConverter.PIL_to_tensor(images=converted),)


class EmbedderNode(BaseNode):
    @classmethod
    def data_dir(cls) -> Path:
        return Path(folder_paths.get_input_directory()) / "data"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        data_dir = cls.data_dir()

        if os.path.exists(data_dir) is False:
            os.makedirs(data_dir)

        data_files = [
            f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
        ]

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "An image to which the information will be embedded (Should be converted to YCbCr)"
                    },
                ),
                "channel": (["Y", "Cb", "Cr"], {}),
                "message_file": (sorted(data_files), {"image_upload": True}),
                "code_words": ("MASK", {}),
                "multiplier": ("INT", {"default": 1, "min": 1, "max": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("embedded",)

    FUNCTION = "embedd"

    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    def embedd(
        self,
        image: list[torch.Tensor],
        channel: list[Literal["Y", "Cb", "Cr"]],
        message_file: list[str],
        code_words: list[torch.Tensor],
        multiplier: list[int],
    ) -> tuple[torch.Tensor]:
        # All inputs are now lists :)
        image: torch.Tensor = image[0]
        channel: Literal["Y", "Cb", "Cr"] = channel[0]
        message_file: str = message_file[0]
        code_words: torch.Tensor = code_words[0]
        multiplier: int = multiplier[0]

        imported_list = ImageConverter.tensor_to_PIL(
            tensor=image,
            mode="YCbCr",
        )

        imported = imported_list[0]  # single image expected

        np_image = numpy.array(imported)

        np_image_ch = np_image.transpose(2, 0, 1)

        channel_idx = {"Y": 0, "Cb": 1, "Cr": 2}[channel]

        # Prepare code words list
        cw_list: list[np.ndarray] = []

        for _, cw in enumerate(code_words):
            word = cw.cpu().numpy() * 255.0
            # Converts to uint8 then to int8 to get negative values
            cw_list.append(word.astype(np.uint8).astype(np.int8))

        if multiplier != 1:
            cw_list = [cw * multiplier for cw in cw_list]

        block_height = cw_list[0].shape[0]
        block_width = cw_list[0].shape[1]

        # Instantiate embedder and embed bits from message
        embedder = MultiEmbedder(
            use_channel=channel_idx,
            block_height=block_height,
            block_width=block_width,
            code_words=cw_list,
        )

        data_generator = BaseEmbedder.file_to_generator(
            file_path=(self.data_dir() / message_file).as_posix(),
        )

        embedded_arr = embedder.embed_info(
            image=np_image_ch.astype(np.uint8),
            data=data_generator,
        )

        if embedded_arr.dtype != np.uint8:
            embedded_arr = embedded_arr.astype(np.uint8)
        pil_out = PIL.Image.fromarray(embedded_arr, mode="YCbCr")
        out_tensor = torch.from_numpy(numpy.array(pil_out) / 255.0)[None,]
        return (out_tensor,)


class PrintInput(BaseNode):
    CATEGORY = "Utils"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "optional": {
                "int_input": ("INT",),
                "float_input": ("FLOAT",),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = tuple()

    FUNCTION = "print_data"

    OUTPUT_NODE = True

    def print_data(self, int_input: int, float_input: float, node_id: str) -> tuple:
        PromptServer.instance.send_progress_text(
            f"PrintInput node received int_input={int_input}, float_input={float_input}",
            node_id=node_id,
        )
        return tuple()


class BaseExtractor:
    def __init__(
        self,
        use_channel: int,
        block_height: int,
        block_width: int,
    ) -> None:
        self.use_channel = use_channel
        self.block_height = block_height
        self.block_width = block_width

    @staticmethod
    def block_generator(
        data: np.ndarray,
        x: int = 8,
        y: int = 8,
    ) -> Generator[np.ndarray, None, None]:
        for i in range(0, data.shape[0], y):
            for j in range(0, data.shape[1], x):
                yield data[i : i + y, j : j + x]

    def extract_info(
        self,
        original_image: np.ndarray,
        modified_image: np.ndarray,
    ) -> Generator[int, None, None]:
        raise NotImplementedError

    @staticmethod
    def bits_to_bytes(bits):
        byte = ""
        count = 0
        for bit in bits:
            byte += str(int(bool(bit)))
            count += 1
            if count == 8:
                yield int(byte, 2)
                byte = ""
                count = 0
        if count > 0:
            byte = byte.ljust(8, "0")
            yield int(byte, 2)


class MultiExtractor(BaseExtractor):
    def __init__(
        self,
        use_channel: int,
        block_height: int,
        block_width: int,
        threshold: float = 0.0,
        expected_elements: list[tuple[int, int]] = [(3, 3)],
    ) -> None:
        super().__init__(use_channel, block_height, block_width)
        self.bits_per_block = len(expected_elements)
        self.threshold = threshold
        self.expected_elements = expected_elements

    @staticmethod
    def dct(block: "np.ndarray") -> "np.ndarray":
        return scipy.fftpack.dct(block.flatten(), norm="ortho").reshape(block.shape)

    @staticmethod
    def idct(block: "np.ndarray") -> "np.ndarray":
        return scipy.fftpack.idct(block.flatten(), norm="ortho").reshape(block.shape)

    def extract_info(
        self,
        original_image: np.ndarray,
        modified_image: np.ndarray,
    ) -> Generator[int, None, None]:
        orig_channel = original_image[self.use_channel]
        mod_channel = modified_image[self.use_channel]

        orig_blocks = list(
            self.block_generator(orig_channel, self.block_width, self.block_height)
        )
        mod_blocks = list(
            self.block_generator(mod_channel, self.block_width, self.block_height)
        )

        for orig_block, mod_block in zip(orig_blocks, mod_blocks):
            if orig_block.shape != (self.block_height, self.block_width):
                continue

            diff = (mod_block.astype(np.int16) - orig_block.astype(np.int16)).astype(
                np.float32
            )

            # inverse transform to get code-word-space differences (assumes embedder used idct/dct accordingly)
            try:
                diff = self.idct(diff)
            except Exception:
                # fallback: use diff as-is
                pass

            for el in self.expected_elements:
                r, c = el
                if r < 0 or c < 0 or r >= diff.shape[0] or c >= diff.shape[1]:
                    continue
                influence = float(diff[r, c])
                if abs(influence) < self.threshold:
                    continue
                bit = 1 if influence > 0 else 0
                yield bit


class ExtractorNode(BaseNode):
    @classmethod
    def data_dir(cls) -> Path:
        return Path(folder_paths.get_input_directory()) / "data"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        data_dir = cls.data_dir()
        if os.path.exists(data_dir) is False:
            os.makedirs(data_dir)
        return {
            "required": {
                "embedded_image": (
                    "IMAGE",
                    {"tooltip": "YCbCr image containing embedded data"},
                ),
                "original_image": (
                    "IMAGE",
                    {
                        "tooltip": "Original YCbCr image before embedding (for comparison)"
                    },
                ),
                "channel": (["Y", "Cb", "Cr"], {}),
                "block_height": ("INT", {"default": 8, "min": 1}),
                "block_width": ("INT", {"default": 8, "min": 1}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "expected_elements": (
                    "STRING",
                    {
                        "default": "[(3,3)]",
                        "tooltip": "Python list of (row,col) positions, e.g. [(3,3),(1,2)]",
                    },
                ),
                "out_filename": ("STRING", {"default": "extracted.bin"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out_filename",)
    FUNCTION = "extract"
    OUTPUT_NODE = True

    def extract(
        self,
        embedded_image: torch.Tensor,
        original_image: torch.Tensor,
        channel: Literal["Y", "Cb", "Cr"],
        block_height: int,
        block_width: int,
        threshold: float,
        expected_elements: str,
        out_filename: str,
    ) -> tuple:
        # Convert embedded image tensor -> PIL -> numpy (0..255)
        embedded_list = ImageConverter.tensor_to_PIL(
            tensor=embedded_image, mode="YCbCr"
        )
        embedded_img = embedded_list[0]
        np_embedded = numpy.array(embedded_img)  # H,W,3
        np_embedded_ch = np_embedded.transpose(2, 0, 1)  # C,H,W

        # Convert original image tensor -> PIL -> numpy (0..255)
        original_list = ImageConverter.tensor_to_PIL(
            tensor=original_image, mode="YCbCr"
        )
        original_img = original_list[0]
        np_original = numpy.array(original_img)  # H,W,3
        np_original_ch = np_original.transpose(2, 0, 1)  # C,H,W

        channel_idx = {"Y": 0, "Cb": 1, "Cr": 2}[channel]

        # parse expected_elements string safely
        try:
            parsed = ast.literal_eval(expected_elements)
            if isinstance(parsed, tuple):
                parsed = [parsed]
            expected_list: list[tuple[int, int]] = list(parsed)
        except Exception:
            raise ValueError(
                "expected_elements must be a python literal like [(3,3),(1,2)]"
            )

        extractor = MultiExtractor(
            use_channel=channel_idx,
            block_height=block_height,
            block_width=block_width,
            threshold=threshold,
            expected_elements=expected_list,
        )

        # Now use the actual original image for proper comparison
        bits_gen = extractor.extract_info(
            original_image=np_original_ch,
            modified_image=np_embedded_ch,
        )

        # convert bits to bytes and write file
        bytes_out = bytes(BaseExtractor.bits_to_bytes(bits_gen))
        out_path = (self.data_dir() / out_filename).as_posix()
        with open(out_path, "wb") as f:
            f.write(bytes_out)

        return (out_path,)


# Compression node (JPEG) - takes in an image, applies JPEG compression
# Returns the compressed image
# Parameters: quality (INT) from 100 to 1


class JPEGCompressionNode(BaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "The image to compress"},
                ),
                "quality": (
                    "INT",
                    {
                        "default": 90,
                        "min": 1,
                        "max": 100,
                        "tooltip": "The quality of the JPEG compression (1-100)",
                    },
                ),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("compressed_image",)
    FUNCTION = "compress"

    def compress(
        self,
        image: torch.Tensor,
        quality: int,
        node_id: str,
    ) -> tuple[torch.Tensor]:
        imported_list = ImageConverter.tensor_to_PIL(
            tensor=image,
        )
        compressed_list = []

        for img in imported_list:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=quality)
            img_bytes.seek(0)
            compressed_img = PIL.Image.open(img_bytes)
            compressed_list.append(compressed_img)

        PromptServer.instance.send_progress_text(
            f"Applied JPEG compression with quality {quality}.",
            node_id=node_id,
        )

        return (ImageConverter.PIL_to_tensor(images=compressed_list),)


# Blur node - takes in an image, applies Gaussian blur
# Returns the blurred image
# Parameters: radius (FLOAT)


class BlurNode(BaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "The image to blur"},
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 20.0,
                        "tooltip": "The radius of the Gaussian blur (in pixels)",
                    },
                ),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blurred_image",)
    FUNCTION = "apply_blur"

    def apply_blur(
        self,
        image: torch.Tensor,
        radius: float,
        node_id: str,
    ) -> tuple[torch.Tensor]:
        imported_list = ImageConverter.tensor_to_PIL(
            tensor=image,
        )
        blurred_list = []

        for img in imported_list:
            blurred_img = img.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
            blurred_list.append(blurred_img)

        PromptServer.instance.send_progress_text(
            f"Applied Gaussian blur with radius {radius}.",
            node_id=node_id,
        )

        return (ImageConverter.PIL_to_tensor(images=blurred_list),)


# Noise Attack Node - takes in an image, adds random noise to it
# (with seed field for reproducibility)
# Returns the noised image


class NoiseAttackNode(BaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "The image to add noise to"},
                ),
                "mode": (
                    [
                        "salt",
                        "pepper",
                    ],
                    {"default": "salt"},
                ),
                "amount": (
                    "FLOAT",
                    {
                        "default": 0.0005,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "The amount of noise to add (as a fraction of pixels)",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noised_image",)
    FUNCTION = "add_noise"

    def add_noise(
        self,
        image: torch.Tensor,
        mode: Literal["salt", "pepper"],
        amount: float,
        seed: int,
        node_id: str,
    ) -> tuple[torch.Tensor]:
        imported_list = ImageConverter.tensor_to_PIL(
            tensor=image,
        )
        noised_list = []

        rng = np.random.default_rng(seed)

        for img in imported_list:
            np_img = np.array(img)

            total_pixels = np_img.shape[0] * np_img.shape[1]
            num_noisy = int(total_pixels * amount)

            coords = [
                (
                    rng.integers(0, np_img.shape[0]),
                    rng.integers(0, np_img.shape[1]),
                )
                for _ in range(num_noisy)
            ]

            for y, x in coords:
                if mode == "salt":
                    np_img[y, x] = 255
                elif mode == "pepper":
                    np_img[y, x] = 0

            noised_list.append(PIL.Image.fromarray(np_img))

        PromptServer.instance.send_progress_text(
            f"Added {amount * 100:.2f}% {mode} noise to image.",
            node_id=node_id,
        )

        return (ImageConverter.PIL_to_tensor(images=noised_list),)


# PSNR node
# Takes in an Original Image and a Test Image
# Calculates the PSNR between the two images
# Returns the PSNR value (FLOAT)
# Prints "progress" to the PromptServer with the PSNR value


class PSNRNode(BaseNode):
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "original_image": (
                    "IMAGE",
                    {"tooltip": "The original image for comparison"},
                ),
                "test_image": (
                    "IMAGE",
                    {"tooltip": "The test image to compare against the original"},
                ),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("psnr_value",)
    FUNCTION = "calculate_psnr"

    def calculate_psnr(
        self,
        original_image: torch.Tensor,
        test_image: torch.Tensor,
        node_id: str,
    ) -> tuple[float]:
        mse = torch.mean((original_image - test_image) ** 2).item()
        if mse == 0:
            psnr = float("inf")
        else:
            max_pixel = 1.0  # assuming images are normalized between 0 and 1
            psnr = 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(
                torch.tensor(mse)
            )
            psnr = psnr.item()

        PromptServer.instance.send_progress_text(
            f"PSNR Value: {psnr:.4f} dB",
            node_id=node_id,
        )

        return (psnr,)


class CompareBinaryDataNode(BaseNode):
    CATEGORY = "Utils"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "original": (
                    "STRING",
                    {
                        "default": "data/data.bin",
                        "tooltip": "Path is relative to the input folder",
                    },
                ),
                "test_data": (
                    "STRING",
                    {
                        "default": "data/extracted.bin",
                        "tooltip": "Path is relative to the input folder",
                    },
                ),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("total_bits", "different_bits", "accuracy_percent")
    FUNCTION = "compare"

    def compare(
        self,
        original: str,
        test_data: str,
        node_id: str,
    ) -> tuple[int, int, float]:
        original = (Path(folder_paths.get_input_directory()) / original).as_posix()
        test_data = Path(test_data).as_posix()
        if not os.path.exists(test_data):
            test_data = (
                Path(folder_paths.get_input_directory()) / test_data
            ).as_posix()

        with open(original, "rb") as f:
            original_bytes = f.read()
        with open(test_data, "rb") as f:
            test_bytes = f.read()[: len(original_bytes)]  # truncate to original length

        total_bits = len(original_bytes) * 8
        different_bits = 0

        for b1, b2 in zip(original_bytes, test_bytes):
            diff = b1 ^ b2
            different_bits += bin(diff).count("1")

        accuracy_percent = (
            (total_bits - different_bits) / total_bits * 100.0
            if total_bits > 0
            else 0.0
        )

        PromptServer.instance.send_sync(
            "stego.compareBinaryData.result",
            {
                "total_bits": total_bits,
                "different_bits": different_bits,
                "accuracy_percent": accuracy_percent,
            },
        )

        PromptServer.instance.send_progress_text(
            f"Total Bits: {total_bits}\nDifferent Bits: {different_bits}\nAccuracy: {accuracy_percent:.4f}%",
            node_id=node_id,
        )

        return (total_bits, different_bits, accuracy_percent)


class MaskBatchNode(BaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple | str]]:
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "mask3": ("MASK",),
                "mask4": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "batch"

    CATEGORY = "Utils"

    def batch(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        mask3: torch.Tensor,
        mask4: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        s = torch.cat((mask1, mask2, mask3, mask4), dim=0)
        return (s,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PrintInput": PrintInput,
    "EmbedderNode": EmbedderNode,
    "ToYCbCr": ToYCbCr,
    "ToRGB": ToRGB,
    "ExtractorNode": ExtractorNode,
    "CompareBinaryDataNode": CompareBinaryDataNode,
    "PSNRNode": PSNRNode,
    "NoiseAttackNode": NoiseAttackNode,
    "JPEGCompressionNode": JPEGCompressionNode,
    "BlurNode": BlurNode,
    "MaskBatchNode": MaskBatchNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PrintInput": "Print Inputs",
    "EmbedderNode": "Embedd stego-info to an image",
    "ToYCbCr": "Convert RGB input to YcBcCr color scheme",
    "ToRGB": "Convert YcBcCr input to RGB color scheme",
    "ExtractorNode": "Extract stego-info from an image",
    "CompareBinaryDataNode": "Compare Binary Data Files",
    "PSNRNode": "Calculate PSNR between two images",
    "NoiseAttackNode": "Add Noise Attack to Image",
    "JPEGCompressionNode": "Apply JPEG Compression to Image",
    "BlurNode": "Apply Blur Effect to Image",
    "MaskBatchNode": "Batch Masks Together",
}

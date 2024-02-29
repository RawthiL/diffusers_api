import base64
from io import BytesIO
from typing import Any, List

from basemodels import ImageData
from PIL import Image


def data_to_images(input_data: ImageData) -> List[Any]:
    # Get all images raw data
    images = list()
    for image_raw_Data in input_data.img_data:
        # Decode
        decoded_data = base64.b64decode(image_raw_Data)
        # Read into image object
        if input_data.img_data_format == "raw_bytes":
            # Read image as raw bytes
            image = Image.frombytes(
                input_data.img_type, (input_data.img_width, input_data.img_height), decoded_data, "raw"
            )
        elif input_data.img_data_format == "file_bytes":
            # Try to open image data as a file
            image = Image.open(BytesIO(decoded_data))
        else:
            raise ValueError("Image format not supported: %s" % input_data.img_data_format)

        images.append(image)

    return images


def images_to_data(output_type: str, images: List[Any]) -> List[str]:
    output_data = list()
    try:
        # Encode all images
        for image in images:
            if output_type == "pil":
                # Return raw bytes
                output_data.append(base64.b64encode(image.tobytes()).decode("utf-8"))
            else:
                # Return an encoded file
                img_io = BytesIO()
                image.save(img_io, format=output_type)
                output_data.append(base64.b64encode(img_io.getvalue()).decode("utf-8"))
    except Exception as e:
        print(e)
        raise ValueError("Cannot convert image to requested format: %s" % output_type)

    return output_data

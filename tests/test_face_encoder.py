from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.face_encoder import load_image, extract_box


@patch('src.face_encoder.cv2.imread')
def test_load_image_valid_path(mock_imread):
    mock_imread.return_value = np.zeros((100, 100, 3), dtype = np.uint8)
    image = load_image('valid_path.jpg')
    assert image.shape == (100, 100, 3)


@patch('src.face_encoder.cv2.imread')
def test_load_image_invalid_path(mock_imread):
    mock_imread.return_value = None
    with pytest.raises(ValueError):
        load_image('invalid_path.jpg')


def test_extract_box_valid_detection():
    detection = MagicMock()
    detection.location_data.relative_bounding_box.xmin = 0.1
    detection.location_data.relative_bounding_box.ymin = 0.2
    detection.location_data.relative_bounding_box.width = 0.3
    detection.location_data.relative_bounding_box.height = 0.4
    image = np.zeros((100, 200, 3), dtype = np.uint8)
    x, y, h, w = extract_box(detection, image)
    assert (x, y, h, w) == (20, 20, 40, 60)


def test_extract_box_zero_size_image():
    detection = MagicMock()
    detection.location_data.relative_bounding_box.xmin = 0.1
    detection.location_data.relative_bounding_box.ymin = 0.2
    detection.location_data.relative_bounding_box.width = 0.3
    detection.location_data.relative_bounding_box.height = 0.4
    image = np.zeros((0, 0, 3), dtype = np.uint8)
    x, y, h, w = extract_box(detection, image)
    assert (x, y, h, w) == (0, 0, 0, 0)

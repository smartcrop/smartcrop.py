import sys

import pytest
from PIL import Image

from smartcrop import cli


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['smartcroppy', *map(str, argv)])
    cli.main()


def test_output_has_the_requested_size(monkeypatch, tmp_path):
    """The crop is written at the requested size, not at a ratio rounded to whole percents"""
    source = tmp_path / 'in.jpg'
    Image.new('RGB', (1920, 1080)).save(source)

    run_cli(monkeypatch, '--width', 300, '--height', 200, source, tmp_path / 'out.jpg')

    with Image.open(tmp_path / 'out.jpg') as output:
        assert output.size == (300, 200)


@pytest.mark.parametrize('option', ['--width', '--height'])
def test_size_under_one_pixel_is_refused(monkeypatch, tmp_path, option):
    """A size under 1 pixel is an argument error, raised before the input is read"""
    with pytest.raises(SystemExit) as exc_info:
        run_cli(monkeypatch, option, 0, tmp_path / 'in.jpg', tmp_path / 'out.jpg')

    assert exc_info.value.code == 2


def test_output_format_follows_the_extension(monkeypatch, tmp_path):
    """An output named .png holds PNG data, not a JPEG stream"""
    source = tmp_path / 'in.jpg'
    Image.new('RGB', (200, 200)).save(source)

    run_cli(monkeypatch, source, tmp_path / 'out.png')

    with Image.open(tmp_path / 'out.png') as output:
        assert output.format == 'PNG'


def test_small_crop_is_not_enlarged(monkeypatch, tmp_path):
    """A crop smaller than the requested size is written as it is, never enlarged"""
    source = tmp_path / 'in.jpg'
    Image.new('RGB', (50, 50)).save(source)

    run_cli(monkeypatch, '--width', 100, '--height', 100, source, tmp_path / 'out.jpg')

    with Image.open(tmp_path / 'out.jpg') as output:
        assert output.size == (50, 50)

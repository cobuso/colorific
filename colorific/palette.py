#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  palette.py
#  palette_detect
#

"""
Detect the main colors used in an image.
"""

from __future__ import print_function

import colorsys
import multiprocessing
import sys
from collections import Counter, namedtuple
from operator import itemgetter, mul, attrgetter

from PIL import Image, ImageChops, ImageDraw
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cmc
from colormath.color_objects import sRGBColor, LabColor

from colorific import config


Color = namedtuple('Color', ['value', 'prominence'])
Palette = namedtuple('Palette', 'colors bgcolor')

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


def color_stream_st(istream=sys.stdin, save_palette=False, **kwargs):
    """
    Read filenames from the input stream and detect their palette.
    """
    for line in istream:
        filename = line.strip()
        try:
            palette = extract_colors(filename, **kwargs)

        except Exception as e:
            print(filename, e, file=sys.stderr)
            continue

        print_colors(filename, palette)
        if save_palette:
            save_palette_as_image(filename, palette)


def color_stream_mt(istream=sys.stdin, n=config.N_PROCESSES, **kwargs):
    """
    Read filenames from the input stream and detect their palette using
    multiple processes.
    """
    queue = multiprocessing.Queue(1000)
    lock = multiprocessing.Lock()

    pool = [multiprocessing.Process(target=color_process, args=(queue, lock),
            kwargs=kwargs) for i in range(n)]
    for p in pool:
        p.start()

    block = []
    for line in istream:
        block.append(line.strip())
        if len(block) == config.BLOCK_SIZE:
            queue.put(block)
            block = []
    if block:
        queue.put(block)

    for i in range(n):
        queue.put(config.SENTINEL)

    for p in pool:
        p.join()


def color_process(queue, lock):
    "Receive filenames and get the colors from their images."
    while True:
        block = queue.get()
        if block == config.SENTINEL:
            break

        for filename in block:
            try:
                palette = extract_colors(filename)
            except:  # TODO: it's too broad exception.
                continue
            lock.acquire()
            try:
                print_colors(filename, palette)
            finally:
                lock.release()


@lru_cache()
def convert_sRGB(c):
    return convert_color(sRGBColor(*c, is_upscaled=True), LabColor)


def distance(c1, c2):
    """
    Calculate the visual distance between the two colors.
    """
    return delta_e_cmc(convert_sRGB(c1), convert_sRGB(c2))


def rgb_to_hex(color):
    return '#%.02x%.02x%.02x' % color


def hex_to_rgb(color):
    assert color.startswith('#') and len(color) == 7
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def extract_colors(
        filename_or_img, min_saturation=config.MIN_SATURATION,
        min_distance=config.MIN_DISTANCE, max_colors=config.MAX_COLORS,
        min_prominence=config.MIN_PROMINENCE, n_quantized=config.N_QUANTIZED):
    """
    Determine what the major colors are in the given image.
    """
    if Image.isImageType(filename_or_img):
        im = filename_or_img
    else:
        temp_image = Image.open(filename_or_img)
        # Workaround for "Too many open files" issue in Pillow
        # see https://github.com/python-pillow/Pillow/issues/1237
        im = temp_image.copy()
        temp_image.close()

    # get point color count
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im = autocrop(im, config.WHITE)  # assume white box
    im = im.convert(
        'P', palette=Image.ADAPTIVE, colors=n_quantized).convert('RGB')
    dist = Counter({color: count for count, color
                    in im.getcolors(n_quantized)})
    n_pixels = mul(*im.size)

    # aggregate colors
    to_canonical = {config.BLACK: config.BLACK}
    aggregated = Counter({config.BLACK: 0})
    sorted_cols = sorted(dist.items(), key=itemgetter(1), reverse=True)
    for c, n in sorted_cols:
        if c in aggregated:
            # exact match!
            aggregated[c] += n
        else:
            d, nearest = min((distance(c, alt), alt) for alt in aggregated)
            if d < min_distance:
                # nearby match
                aggregated[nearest] += n
                to_canonical[c] = nearest
            else:
                # no nearby match
                aggregated[c] = n
                to_canonical[c] = c

    # order by prominence
    colors = sorted(
        [Color(c, n / float(n_pixels)) for c, n in aggregated.items()],
        key=attrgetter('prominence'), reverse=True)

    colors, bg_color = detect_background(im, colors, to_canonical)

    # keep any color which meets the minimum saturation
    sat_colors = [c for c in colors if meets_min_saturation(c, min_saturation)]
    if bg_color and not meets_min_saturation(bg_color, min_saturation):
        bg_color = None
    if sat_colors:
        colors = sat_colors
    else:
        # keep at least one color
        colors = colors[:1]

    # keep any color that hits the min prominence value
    # only colours sufficiently different from the background color will be kept
    color_list = []
    color_count = 0

    for color in colors:

        similar_to_bg = bg_color and (distance(color.value, bg_color.value) < min_distance)
        if color.prominence > min_prominence and not similar_to_bg:
            color_list.append(color)
            color_count += 1

        if color_count >= max_colors:
            break

    return Palette(color_list, bg_color)


def norm_color(c):
    r, g, b = c
    return r / 255.0, g / 255.0, b / 255.0


def detect_background(im, colors, to_canonical):


    # work out the background color - first, sample around the edges
    w, h = im.size
    width_from_edge = 10
    points = [
        (width_from_edge, width_from_edge), (width_from_edge, h / 2),
        (width_from_edge, h - width_from_edge), (w / 2, h - width_from_edge),
        (w - width_from_edge, h - width_from_edge), (w - width_from_edge, h / 2),
        (w - width_from_edge, 0), (w / 2, width_from_edge)
    ]
    edge_dist = Counter(im.getpixel(p) for p in points)

    # aggregate background colours, to ensure that background is respected even if it's slightly inconsistent
    min_bg_distance = 5.0
    aggregated = Counter()
    for c, n in edge_dist.items():
        if c in aggregated:
            aggregated[c] += n
        else:
            if len(aggregated) > 0:
                d, nearest = min((distance(c, alt), alt) for alt in aggregated)
                if d < min_bg_distance:
                    # close match
                    aggregated[nearest] += n
                else:
                    # no close match
                    aggregated[c] = n
            else:
                aggregated[c] = n


    # next, sample in the centre (40%, 40%)
    points = [
        (4.5 * w / 10.0, 4.5 * h / 10.0), 
        (5.5 * w / 10.0, 4.5 * h / 10.0),
        (4.5 * w / 10.0, 5.5 * h / 10.0),
        (5.5 * w / 10.0, 5.5 * h / 10.0),
    ]
    centre_dist = Counter(im.getpixel(p) for p in points)

    (majority_col, majority_count), = aggregated.most_common(1)
    (centre_col, centre_count), = centre_dist.most_common(1)

    min_distance = 5
    if majority_count >= 3 and distance(centre_col, majority_col) > min_distance:
        # we have a background color, but only if three of the sampled edge points
        # are the same colour, and the edge points are sufficiently different from
        # the central points
        canonical_bg = to_canonical[majority_col]
        bg_color, = [c for c in colors if c.value == canonical_bg]
        colors = [c for c in colors if c.value != canonical_bg]
    else:
        # no background color
        bg_color = None

    return colors, bg_color


def print_colors(filename, palette):
    colors = '%s\t%s\t (BG: %s)' % (
        filename, ','.join(rgb_to_hex(c.value) for c in palette.colors),
        palette.bgcolor and rgb_to_hex(palette.bgcolor.value) or '')
    print(colors)
    sys.stdout.flush()


def save_palette_as_image(filename, palette):
    "Save palette as a PNG with labeled, colored blocks"
    output_filename = '%s_palette.png' % filename[:filename.rfind('.')]
    size = (80 * len(palette.colors), 80)
    im = Image.new('RGB', size)
    draw = ImageDraw.Draw(im)
    for i, c in enumerate(palette.colors):
        v = colorsys.rgb_to_hsv(*norm_color(c.value))[2]
        (x1, y1) = (i * 80, 0)
        (x2, y2) = ((i + 1) * 80 - 1, 79)
        draw.rectangle([(x1, y1), (x2, y2)], fill=c.value)
        if v < 0.6:
            # white with shadow
            draw.text((x1 + 4, y1 + 4), rgb_to_hex(c.value), (90, 90, 90))
            draw.text((x1 + 3, y1 + 3), rgb_to_hex(c.value))
        else:
            # dark with bright "shadow"
            draw.text((x1 + 4, y1 + 4), rgb_to_hex(c.value), (230, 230, 230))
            draw.text((x1 + 3, y1 + 3), rgb_to_hex(c.value), (0, 0, 0))

    im.save(output_filename, "PNG")


def meets_min_saturation(c, threshold):
    return colorsys.rgb_to_hsv(*norm_color(c.value))[1] >= threshold


def autocrop(im, bgcolor):
    "Crop away a border of the given background color."
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, bgcolor)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

    return im  # no contents, don't crop to nothing

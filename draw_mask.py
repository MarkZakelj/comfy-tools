from PIL import Image, ImageDraw


def draw_lines(points: list[list[tuple[int]]], width: int, height: int):
    """
    Draw lines on an image
    :param points: in the form [[(x1, y1), (x2, y2)], ...]
    :param width: canvas width
    :param height: canvas height
    :return: image with points drawn on them
    """
    img = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img)

    for p1, p2 in points:
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill='white', width=6)

    return img


def main():
    pass


if __name__ == '__main__':
    main()

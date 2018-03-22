import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default='gymnastics.jpg', help='the image to segment')
    args = parser.parse_args()
    print('SOURCE IMAGE:', args.image)

import argparse
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--username", required=False, default="doug", help="The Users user name...")
ap.add_argument("-s", "--start", required=False, help="Image to start benchmarking run")
ap.add_argument("-e", "--end", required=False, help="Image to end benchmarking run")

root = '../../'

args = vars(ap.parse_args())

username = args["username"]
start = -1
end = -1

if args["start"] is not None:
    start = int(args["start"])

if args["end"] is not None:
    end = int(args["end"])

path = root + 'images/' + str(username) + '/'

directories = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

values = []

findNumer = re.compile(r'(\d+)\/$')

for directory in directories:
    hsvCount_path = directory + '/steps/4.csv'
    imageStat_path = directory + '/reference/imageStats.csv'

    imageNumber = int(re.compile(r'(\d+)$').search(directory).group(1))

    if (start != -1 and imageNumber < start):
        continue

    if (end != -1 and imageNumber > end):
        continue


    if os.path.exists(hsvCount_path):
        os.remove(hsvCount_path)

    if os.path.exists(imageStat_path):
        os.remove(imageStat_path)


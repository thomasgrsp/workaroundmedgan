import argparse

def parse_arguments(parser):

	parser.add_argument('--tuple', help="Coordinate", dest="cord", type=coords)
	args = parser.parse_args()
	return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)

print(args.cord)
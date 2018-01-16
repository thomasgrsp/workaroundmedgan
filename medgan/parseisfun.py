import argparse

def parse_arguments(parser):
	def parse_dims(s):
		dims = s.split(',')
		return tuple(map(int, dims))
	parser.add_argument('--tuple', help="Coordinate", dest="cord", type=coords)
	args = parser.parse_args()
	return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)

print(args.cord)
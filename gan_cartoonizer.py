
import argparse
import matplotlib.pyplot as plt

from gan_cartoon_animator import Generator

class Cartoonizer():
	
	def main(self):
		if args.input_file is None:
			print("Must provide a valid input image file to cartoonize.")
			return

		if args.animate:
			# Animate the output cartoon image
			if args.video_sample is not None:
				# animate to the video sample
				pass
			else:
				# animate to default video
				pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", help="Input file to turn into a cartoon animation. Must be > 256x256.", nargs="?")
	parser.add_argument("--animate", help="Whether or not the cartoon image will be turned into an animation.", nargs="?", default=False)
	parser.add_argument("--video_sample", help="Video sample (.mp4) to animate the cartoon face image to.", nargs="?", default="")
	args = parser.parse_args()

	Cartoonizer = Cartoonizer()
	Cartoonizer.main()

	from PIL import Image
	img = Image.open('./raw/portrait.png')
	# img = img.resize((4200, 4200), Image.ANTIALIAS)
	# print(img.size)
	# img.show()
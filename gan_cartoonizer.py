	import argparse
	import numpy as np
	import matplotlib.pyplot as plt
	import torch
	import torch.nn
	import torchvision.utils as vutils
	from PIL import Image
	import glob
	import os, os.path
	# import cv2

	from gan_facegenerator import Generator

	# REQUIRES GPU TO RUN. CANNOT LOAD MODELS -> DEPLOY TO GPU SERVER.

	# import pretrained_networks
	# blended_url = "https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU" 
	# ffhq_url = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl"

	# _, _, G_blended = pretrained_networks.load_networks(blended_url) # load cartoon generator model
	# _, _, G = pretrained_networks.load_networks(ffhq_url)

	# Class function:
	# Create

	class Cartoonizer():

		def __init__(self):
			self.nz = 100
			self.IMG_SIZE = 128

			self.device = ("cuda:0" if torch.cuda.is_available() else "cpu" )
			self.netG = Generator().to(self.device)

		def main(self):
			if args.mode == "facegen":
				if args.model_path is None:
					print("You must specify a valid model file (/models).")
					return
				self.generate_face(model_path=args.model_path)

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

		def generate_face(self, model_path):
			print("generating face...")
			self.netG.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
			noise = torch.randn(1, self.nz, 1, 1, device=self.device)
			face = self.netG(noise)[0].detach().cpu().permute(1,2,0)
			face = vutils.make_grid(face, padding=2, normalize=True)
			plt.imshow(face)
			plt.show()

		def unpack_bz2(src_path):
			data = bz2.BZ2File(src_path).read()
			dst_path = src_path[:-4]
			with open(dst_path, 'wb') as fp:
				fp.write(data)
			return dst_path

		def cartoonize_face(self, raw_dir, align_dir):
			for f in os.listdir(raw_dir):
				img = Image.open(os.path.join(raw_dir, f))
				# img.resize((256, 256), Image.ANTIALIAS)
				img = img.resize((333, 333))
				width, height = img.size
				cX = width/2
				cY = height/2

				left = cX - 128
				top = cY - 128
				right = cX + 128
				bottom = cY + 128

				img = img.crop((left, top, right, bottom))
				img.show()
				arr = np.array(img)
				print(arr)

		def cartoonize_face(self, source_img, display_image=False):
			# if display_image:
			#     source_img = IPImage(filename=source_img, width=256)
			#     display(source_img)

			%cd /content/stylegan2
			# !python align_images.py raw aligned
			# project face image into n dim latent vector (z)
			!python project_images.py --num-steps 500 aligned generated

			latent_dir = Path("generated")
			latents = latent_dir.glob("*.npy")
			for latent_file in latents:
				# pass latent vector generated from face image through stylegan2 Generator network.
				latent = np.load(latent_file)
				latent = np.expand_dims(latent, axis=0)
				synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
				images = self.G_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
				PILImage.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}-cartoon.jpg"))

        cartoon_img = IPImage(filename="/content/stylegan2/generated/face_aligned-cartoon.jpg", width=256)
        if display_image:
            display(cartoon_img)

		def generate_cartoonize_face(self, model_path, raw_face_path, aligned_dir, display_image=False):
			# generate netG face, saves as raw/raw_face.jpg.
			self.generate_face(model_path, display_image=display_image)
			self.resize_face((256,256), raw_face_path, aligned_dir)
			self.cartoonize_face(source_img=f"{aligned_dir}/face_aligned.jpg", display_image=True)


	if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument("--mode", help="facegen | facegen_cartoon | animate_cartoon")
		parser.add_argument("--model_path", help="Model to use to generate an artificial human face. Find in e-Dylan/gan_cartoonizer/models", nargs="?", default="models/netG_EPOCHS=12_IMGSIZE=128.pth")
		parser.add_argument("--input_file", help="Input file to turn into a cartoon animation. Must be >= 256x256.", nargs="?")
		parser.add_argument("--animate", help="Whether or not the cartoon image will be turned into an animation.", nargs="?", default=False)
		parser.add_argument("--video_sample", help="Video sample (.mp4) to animate the cartoon face image to.", nargs="?", default="")
		args = parser.parse_args()

		Cartoonizer = Cartoonizer()
		# Cartoonizer.main()
		Cartoonizer.cartoonize_face("C:/Users/Dylan/Desktop/build/python_projects/gan_cartoonizer/raw", "C:/Users/Dylan/Desktop/build/python_projects/gan_cartoonizer/aligned")
		# Cartoonizer.generate_face()

		# from PIL import Image
		# img = Image.open('./raw/portrait.png')
		# img = img.resize((4200, 4200), Image.ANTIALIAS)
		# print(img.size)
		# img.show()
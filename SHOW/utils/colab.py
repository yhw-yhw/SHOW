from IPython.display import HTML
from base64 import b64encode
from IPython.display import Image


def show_im():
  from IPython.display import Image, display
  import tempfile
  import os.path as osp
  with tempfile.TemporaryDirectory() as tmpdir:
      file_name = osp.join(tmpdir, 'pose_results.png')
      cv2.imwrite(file_name, vis_result)
      display(Image(file_name))

def video(path):
  mp4 = open(path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML('<video width=500 controls loop> <source src="%s" type="video/mp4"></video>' % data_url)

if __name__ == '__main__':
    video('output/dancer/dancer_result.mp4')
    Image(filename='output/coco_images/coco_images_output/COCO_val2014_000000004700.png')
    
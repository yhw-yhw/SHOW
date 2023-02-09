import openpifpaf
import matplotlib.pyplot as plt
import PIL
import json
import cv2

class pifpaf_detector(object):
    # chk_type:str = 'shufflenetv2k30-wholebody'
    
    # def __post_init__(self):
    def __init__(self):
        chk_type:str = 'shufflenetv2k30-wholebody'
        self.predictor = openpifpaf.Predictor(checkpoint=chk_type)
        
    def process(self,full_file_name,out_file_name,out_img_name):
        pil_im = PIL.Image.open(full_file_name).convert('RGB')
        predictions, _, _ = self.predictor.pil_image(pil_im)

        # sample_img=cv2.cvtColor(sample_img,cv2.COLOR_BGR2RGB)
        # pil_im = PIL.Image.fromarray(sample_img)
        # predictions, gt_anns, image_meta = self.pose_predictor.pil_image(pil_im)
        # ret = [i.json_data() for i in predictions]
        
        with open(out_file_name, "w") as fp:
            r = [i.json_data() for i in predictions]
            json.dump(r, fp)
            
        self.save_vis(pil_im,predictions,out_img_name)

    def save_vis(self,pil_im,predictions,out_img_name):
        annotation_painter = openpifpaf.show.AnnotationPainter()
        with openpifpaf.show.image_canvas(pil_im) as ax:
            annotation_painter.annotations(ax, predictions)
            ax.figure.savefig(out_img_name)

    def predict( self, sample_img):

        # self.pose_ret_list.append(ret)
        annotation_painter = openpifpaf.show.AnnotationPainter()
        with openpifpaf.show.image_canvas(pil_im) as ax:
            annotation_painter.annotations(ax, predictions)
            canvas = FigureCanvasAgg(ax.figure)
            canvas.draw()
            buf = canvas.buffer_rgba()
            X = np.asarray(buf)
            X=cv2.cvtColor(X,cv2.COLOR_RGBA2RGB)
            self.pose_ret_img=cv2.resize(X,self.predict_size)
            # PIL.Image.fromarray(X).show()         
            # fig.set_tight_layout(True)
            # ax.figure.savefig('./test.jpg')
            # import pdb;pdb.set_trace()
        return ret,self.pose_ret_img
    
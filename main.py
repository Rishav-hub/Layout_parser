import cv2
import layoutparser as lp
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
class Detector:
    def __init__(self, filename):
        self.filename = filename
        self.configPath = 'models/config.yml'
        self.modelPath = 'models/model_final.pth'
        self.output_filename = 'output.jpg'
    
    def detect_layout(self):
        image = cv2.imread(self.filename)
        image = image[..., ::-1]

        # Load model

        model = lp.Detectron2LayoutModel(self.configPath, self.modelPath, 
                                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        
        layout_boxes = model.detect(image)

        detected_image = lp.draw_box(image, layout_boxes, box_width=3, show_element_type= True)

        cv2.imwrite(self.output_filename, cv2.cvtColor(np.array(detected_image), cv2.COLOR_BGR2RGB))
        return layout_boxes, detected_image
    
    def extract_text(self):
        image = cv2.imread(self.filename)
        image = image[..., ::-1]
        ocr_agent = lp.TesseractAgent(languages='eng')
        layout_boxes, detected_image = self.detect_layout()
        text_blocks = lp.Layout([b for b in layout_boxes if b.type=='Text'])
        figure_blocks = lp.Layout([b for b in layout_boxes if b.type=='Figure'])

        text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

        for block in text_blocks:
            segment_image = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(image))
        # add padding in each image segment can help
        # improve robustness

            text = ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)        

        with open('random.txt', 'w+') as f:
    
            for txt in text_blocks:
                f.write('\n-------------------------\n')
                f.write(txt.text)
                        



if __name__ == "__main__":
    clApp = Detector('InputImage.JPG')
    clApp.detect_layout()
    clApp.extract_text()


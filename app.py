import gradio as gr
import fastbook as fb
import numpy as np
from PIL import Image

def classify_image(image):
    print(type(image))
    print(image)

    img = Image.fromarray(image)
    print(img)

    img.resize((128,128))

    pred, idx, probs = learn_inf.predict(image)
    return dict(zip(learn_inf.dls.vocab, map(float, probs)))

path = fb.Path()

learn_inf = fb.load_learner(path/'export.pkl')
print(learn_inf.dls.vocab)

image = gr.Image()
label = gr.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(inline=False)
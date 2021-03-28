from hedgiefinder.inference import predict
import hedgiefinder
import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

imdir = Path(r'data/test/images')
imlist = list(imdir.rglob('*.png'))


def main():
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(imlist) - 1, 0)
    show_mask = False
    if st.button("Mask ON"):
        show_mask = True

    if st.button("Mask OFF"):
        show_mask = False

    predict = False
    if st.button("Predict"):
        predict = True

    if show_mask:
        st.text("Mask On")
    else:
        st.text("Mask Off")
    show_image(imlist[selected_frame_index], show_mask, predict)


def show_image(fname, mask=False, predict=False):
    img = np.array(Image.open(fname)).astype(np.uint8)

    if mask or predict:
        msk = np.array(hedgiefinder.dataloading.get_msk(fname))
        img1 = hedgiefinder.inference.alpha_mask(img, msk)
    else:
        img1 = img

    st.image(img1, use_column_width=True)

    if predict:
        hf = hedgiefinder.HedgieFinder(fname, cleanup=False).predict([fname])
        img2 = hedgiefinder.inference.alpha_mask(hf.originals, hf.predictions)
        st.image(img2, use_column_width=True)


if __name__ == '__main__':
    main()
import streamlit as st
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO

st.title("Basic Image Operations")

uploaded_file = st.file_uploader("Choose a image file", type=["jpg","jpeg","png"])

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    opencv_image_g = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    h,w = opencv_image_g.shape


    # st.write(h,w)
    select = st.sidebar.selectbox("Process",["Deblurr","Detection","Cannize"])


    if select == "Deblurr":
        t1 = st.sidebar.slider("T1",2,7)
        t2 = st.sidebar.slider("T2",2,7)
        t3 = st.sidebar.slider("T3",1,5)
        t4 = st.sidebar.slider("T4",7,15)
        dst = cv2.fastNlMeansDenoisingColored(opencv_image,None,t1,t2,t3,t4)
        st.image(dst, channels="BGR")
        result = Image.fromarray(dst)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)

    # elif select == "Gradient Filter":
    #     gfil = st.sidebar.radio("Select the type of filter",("laplacian","sobelx","sobely"))
    #     laplacian = cv2.Laplacian(opencv_image_g,cv2.CV_64F)
    #     laplacian = np.int0(laplacian)
    #     sobelx = cv2.Sobel(opencv_image_g,cv2.CV_64F,1,0,ksize=5)
    #     sobelx = np.int0(sobelx)
    #     sobely = cv2.Sobel(opencv_image_g,cv2.CV_64F,0,1,ksize=5)
    #     if gfil == "laplacian":
    #         st.image(sobelx)
    if select == "Cannize":
        min = st.sidebar.slider("minVal",0,500)
        max = st.sidebar.slider("maxVal",0,500)
        edges = cv2.Canny(opencv_image_g,min,max)
        st.image(edges)
        result = Image.fromarray(edges)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)

    else:
        sub = st.sidebar.checkbox("Eye Detection")
        faces = face_cascade.detectMultiScale(opencv_image_g,1.3,5)
        i = 0
        for (x,y,w,h) in faces:
            img = cv2.rectangle(opencv_image,(x,y),(x+w,y+h),(0,0,255),5)
            i+=1
            gray = opencv_image_g[y:y+h,x:x+w]
            color = opencv_image[y:y+h,x:x+w]
            if sub:
                eyes = eye_cascade.detectMultiScale(gray,1.3,5)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(color, (ex,ey), (ex+ew,ey+eh),(0,255,0),4)

        st.image(opencv_image, channels="BGR")

        if i:
            st.write(f"There are {i} people in this image")




    # opencv_image = cv2.line(opencv_image,(0,0),(w,h),(255,0,0),5)
    #
    # st.image(opencv_image, channels="BGR")

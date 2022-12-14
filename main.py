#! /bin/env python3
from fastapi import FastAPI,Form,File
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import tensorly as tl
from scipy.io import loadmat
from scipy.linalg import svd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import io
import mtcnn
import PIL.Image as Image
import os
import pickle
from pydantic import BaseModel
# import matplotlib.pyplot as plt
filename = 'finalized_model.pkl'
filename2 = 'matrix.pkl'
model = pickle.load(open(filename, 'rb'))
projected_matrix = pickle.load(open(filename2, 'rb'))
app = FastAPI()
Categories=['pain','neutral']
detector = mtcnn.MTCNN()
def getRecFace(img):
  face=detector.detect_faces(img)
  left_eye=face[0]['keypoints']['left_eye']
  right_eye=face[0]['keypoints']['right_eye']
  mouth_left=face[0]['keypoints']['mouth_left']
  mouth_right=face[0]['keypoints']['mouth_right']
  RecFace=img[min(left_eye[1],right_eye[1])-120:max(mouth_left[1],mouth_right[1])+60,left_eye[0]-60:right_eye[0]+60]
  return RecFace
origins = [
    "http://localhost:3000",
    "localhost:3000"
]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}
@app.post("/predict")
async def  setpic( pic:bytes = File(...)):
    print('testx')
    # print(pic)

    image = io.BytesIO(pic)
    open('test.png','wb+').write(pic)
    imagee = imread('./test.png')
    # response = await store_profile_image(form, str(base64.b64encode(form)))
    # return image
    img=getRecFace(imagee[:,:,:3])
    img=resize(img,(100,100,3))
    projected_data = tl.tenalg.multi_mode_dot(img,projected_matrix,transpose=True)
    _input=projected_data.reshape(1,546)
    # pred_idx = model.predict_proba(_input)[0]
    proba = model.predict_proba(_input)
    result='neutral'
    if(proba[0][0]>proba[0][1]):
        result='pain'
    return {'pain':proba[0][0],'neutral':proba[0][1],'result':result}
    # return {'prediction': Categories[pred_idx]}

if __name__ == "__main__":
  port = int(os.environ.get('PORT', 5000))
  run(app, host="0.0.0.0", port=port)

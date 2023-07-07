from fastapi import FastAPI, File, UploadFile
from image_preprocessing import RequestImageConverter
from image_preprocessing import TextRecognizer
from inferencing import TFLiteInferencer

app = FastAPI()

@app.get("/")
async def root():
    return "Hello World"

@app.post('/predict')
async def create_upload_file(image: UploadFile = File(...)):
    img_file = await image.read()

    if not img_file:
        return {"result_predict" : ""}
    converter = RequestImageConverter(img_file)
    converted_image = converter.convert()

    inferencer = TFLiteInferencer(converted_image)
    prediction = inferencer.predict()

    return {"result_predict" : prediction}


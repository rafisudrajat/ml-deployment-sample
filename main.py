from typing import Union
from model import SimpleCNN
from fastapi import FastAPI, UploadFile, Depends
import inference
from typing_extensions import Annotated

app = FastAPI()


@app.get("/binary-classification")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile
                            #  model:Annotated[dict, Depends(SimpleCNN)]
                             ):
    print("file type", type(file.file))
    return {"filename": file.filename}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
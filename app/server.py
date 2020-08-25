from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision import load_learner, open_image, sys, Path
from gradcam import GradCam
import base64
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

export_file_url = 'https://drive.google.com/uc?export=download&id=1U7yj3i7lb3ZEFrN06vBItv97Mrxa174B'
export_file_name = 'export.pkl'

classes = ['Consolidation', 'Edema', 'Pleural Effusion']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))
app.mount('/tmp', StaticFiles(directory='/tmp'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_detector():
    await download_file(export_file_url, path/export_file_name)
    # build the model from a config file and a checkpoint file
    model = init_detector('/mmdetection/configs/rsna/retinanet_r101_fpn_rsna.py', '/mmdetection/checkpoints/rsna.pth', device='cuda:0')



loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_detector())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    # bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(img_bytes)

def predict_from_bytes(bytes):
    # test a single image
    result = inference_detector(model, BytesIO(bytes))
    # show the results
    show_result_pyplot(model, BytesIO(bytes), result, score_thr=0.0, out_file='/tmp/out_file.png')
   
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    
    result_html = str(result_html1.open().read() +rs + result_html2.open().read())
    return HTMLResponse(result_html)
    # return HTMLResponse("Bingo")

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=9999)

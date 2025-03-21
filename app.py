from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# モデル読み込み（30日後予測モデル）
model_path = os.path.join(os.path.dirname(__file__), "stock_predict_30_ahead.keras")
model = tf.keras.models.load_model(model_path)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, code: str = Form(...)):
    try:
        data = yf.download(code, start='2010-01-01')
        if data.empty:
            return templates.TemplateResponse("index.html", {"request": request, "error": "株価データが取得できませんでした"})

        close_prices = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # 直近180日間のデータを使用
        if len(scaled_data) < 180:
            return templates.TemplateResponse("index.html", {"request": request, "error": "データが180日分未満です"})

        X_test = [scaled_data[-180:, 0]]
        X_test = np.array(X_test).reshape((1, 180, 1))
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        # グラフ作成
        fig, ax = plt.subplots()
        ax.plot(close_prices[-180:].index, close_prices[-180:].values, label='過去180日終値')
        ax.axhline(predicted_price, color='r', linestyle='--', label='30日後予測')
        ax.set_title(f"{code}：30日後の株価予測")
        ax.legend()
        fig.autofmt_xdate()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "predicted_price": predicted_price,
            "graph": image_base64
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"エラー: {str(e)}"})
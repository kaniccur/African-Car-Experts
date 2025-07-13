
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
from bs4 import BeautifulSoup
import torch
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Car(BaseModel):
    id: int
    name: str
    price: str
    image: str
    location: str
    details: str

mock_cars = [
    Car(id=1, name="Toyota Corolla 2018", price="$12,000", image="https://images.unsplash.com/photo-1592194996308-7b43878e84a6", location="Nairobi, Kenya", details="1.8L engine, automatic, 65,000km, silver"),
    Car(id=2, name="Honda Fit 2015", price="$6,500", image="https://images.unsplash.com/photo-1616271221449-e3b42c564b53", location="Mombasa, Kenya", details="1.3L hybrid, automatic, 80,000km, blue"),
]

def scrape_jiji(query: str) -> List[Car]:
    url = f"https://jiji.co.ke/search?query={query}"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.select(".b-list-advert__item")
        results = []
        for idx, item in enumerate(items[:6]):
            title = item.select_one(".b-list-advert__title").text.strip()
            price = item.select_one(".b-list-advert__price").text.strip()
            image = item.select_one("img")
            img_url = image["data-src"] if image and "data-src" in image.attrs else ""
            location = item.select_one(".b-advert__region")
            location = location.text.strip() if location else "Kenya"
            results.append(Car(id=100 + idx, name=title, price=price, image=img_url, location=location, details="From Jiji.co.ke"))
        return results
    except:
        return []

model = models.resnet18(pretrained=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.strip().split("\n")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>AI Car Finder</title>
        <style>
            body { font-family: sans-serif; padding: 2rem; max-width: 800px; margin: auto; }
            .car { border: 1px solid #ccc; margin: 1rem 0; padding: 1rem; border-radius: 8px; }
            img { width: 100%; max-height: 200px; object-fit: cover; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>AI Car Finder</h1>
        <form action="/search" method="get">
            <input name="q" placeholder="Search e.g. Toyota" />
            <button type="submit">Search</button>
        </form>
        <br />
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" />
            <button type="submit">Detect Car from Image</button>
        </form>
    </body>
    </html>
    """

@app.get("/search", response_class=HTMLResponse)
def search_page(q: str = Query("")):
    cars = scrape_jiji(q) or [car for car in mock_cars if q.lower() in car.name.lower()]
    html = f"<h2>Results for '{q}'</h2>"
    for car in cars:
        html += f"<div class='car'><img src='{car.image}' /><h3>{car.name}</h3><p>{car.price}</p><p>{car.location}</p><p>{car.details}</p></div>"
    html += "<a href='/'>Back</a>"
    return HTMLResponse(content=html)

@app.post("/predict")
def predict_car(image: UploadFile = File(...)):
    try:
        img_bytes = image.file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)
            label = labels[predicted.item()]
        return HTMLResponse(f"<h2>Detected: {label}</h2><a href='/'>Back</a>")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

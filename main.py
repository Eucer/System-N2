from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bson import ObjectId
import math
import uvicorn

app = FastAPI()

# CORS middleware settings
origins = [
    "http://localhost:5173",  
    "https://www.douvery.com",  
   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient("mongodb+srv://germys:LWBVI45dp8jAIywv@douvery.0oma0vw.mongodb.net/Production")
db = client["Production"]
collection = db["products"]

# Leer los datos y almacenarlos en un DataFrame
product_data = pd.DataFrame(list(collection.find()))
product_data['_id'] = product_data['_id'].apply(lambda x: str(x))
product_data.drop('_id', axis=1, inplace=True, errors='raise')


# Agregar una columna "full_text" que contenga información adicional sobre el producto
product_data['full_text'] = product_data['category'] + ' ' + product_data['subCategory'] + ' ' + product_data['name']

# Crear un vectorizador para convertir las cadenas de texto a vectores numéricos
vectorizer = CountVectorizer(stop_words='english')
product_vectors = vectorizer.fit_transform(product_data['full_text'])

# Calcular la similitud entre los productos basada en los vectores numéricos de sus descripciones
similarities = cosine_similarity(product_vectors)


def convert_to_serializable(value):
    # Convierte el ObjectId en str
    if isinstance(value, ObjectId):
        return str(value)
    # Recursivamente convierte cualquier ObjectId anidado en el diccionario
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}
    # Recursivamente convierte cualquier ObjectId anidado en la lista
    elif isinstance(value, list):
        return [convert_to_serializable(v) for v in value]
    # Convierte los valores de float fuera de rango en None
    elif isinstance(value, float) and (value == float('inf') or value == float('-inf') or math.isnan(value)):
        return None
    # Devuelve otros valores como están
    return value


def products_to_json(products):
    # Convierte cada producto en un diccionario serializable
    serializable_products = [
        {key: convert_to_serializable(value) for key, value in product.items()}
        for product in products
    ]
    # Serializa la lista a JSON
    return serializable_products


# Función para recomendar productos similares
def recommend_products(product_id):
    # Obtiene la fila del producto de entrada
    product_row = product_data[product_data['dui'] == product_id].iloc[0]

    # Calcula la similitud del producto de entrada con todos los demás productos
    product_similarities = similarities[product_row.name]

    # Obtiene los índices de los productos más similares
    closest_product_indices = product_similarities.argsort()[::-1][1:11]

    # Obtiene los nombres de los productos más similares
    closest_products = product_data.iloc[closest_product_indices]

    # Convierte los ObjectId en el diccionario de productos y serializa a JSON
    return products_to_json(closest_products.to_dict('records'))



@app.get("/", tags=["Root"])
async def read_root():
  return { 
    "message": "Welcome to my notes application, use the /docs route to proceed"
   }

# Definir la ruta para la recomendación de productos similares
@app.get("/recommend_products/{product_id}")
def get_recommendations(product_id: str):
    recommendations = recommend_products(product_id)
    return recommendations


if __name__ == "__main__":
  uvicorn.run("server.api:app", host="0.0.0.0", port=8000, reload=True)

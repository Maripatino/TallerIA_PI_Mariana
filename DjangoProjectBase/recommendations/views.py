from django.shortcuts import render
from django.http import HttpResponse
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

def get_embedding(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommendations_view(request):
    recommended_movies = []
    error = None
    prompt = ""

    if request.method == 'POST':
        prompt = request.POST.get('prompt', '').strip()

        if prompt:
            try:
                # Cargar configuración de OpenAI
                env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'openAI.env')
                load_dotenv(env_path)

                api_key = os.environ.get('openai_apikey') or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    error = "Error: Clave de OpenAI no configurada."
                    return render(request, 'recommendations.html', {
                        'recommended_movie': recommended_movie,
                        'error': error,
                        'prompt': prompt
                    })

                client = OpenAI(api_key=api_key)

                # Generar embedding del prompt
                prompt_emb = get_embedding(prompt, client)

                # Obtener todas las películas con embeddings
                movies = Movie.objects.exclude(emb__isnull=True).exclude(emb=b'')

                if not movies:
                    error = "No hay películas con embeddings disponibles. Ejecuta 'python manage.py movie_embeddings' primero."
                    return render(request, 'recommendations.html', {
                        'recommended_movie': recommended_movie,
                        'error': error,
                        'prompt': prompt
                    })

                # Calcular similitudes
                similarities = []
                for movie in movies:
                    try:
                        movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                        sim = cosine_similarity(prompt_emb, movie_emb)
                        similarities.append((movie, sim))
                    except Exception as e:
                        continue  # Saltar películas con embeddings corruptos

                if not similarities:
                    error = "Error al procesar embeddings de películas."
                    return render(request, 'recommendations.html', {
                        'recommended_movies': [],
                        'error': error,
                        'prompt': prompt
                    })

                # Encontrar las películas con mayor similitud (top 3)
                similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
                recommended_movies = similarities[:3]

            except Exception as e:
                error = f"Error al procesar la recomendación: {str(e)}"

    return render(request, 'recommendations.html', {
        'recommended_movies': recommended_movies,
        'error': error,
        'prompt': prompt
    })

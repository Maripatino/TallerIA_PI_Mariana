import os
import random
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate embeddings for all movies and display a random movie's embedding"

    def add_arguments(self, parser):
        parser.add_argument(
            '--show-random',
            action='store_true',
            help='Show embeddings of a random movie after generation',
        )

    def handle(self, *args, **kwargs):
        # ✅ Load OpenAI API key desde openAI.env en el directorio base del proyecto
        # Usa multiple paths para asegurar que encuentra el archivo
        possible_paths = [
            'openAI.env',  # Desde el directorio actual (DjangoProjectBase)
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'openAI.env'),  # Ruta desde commands
        ]
        
        env_path = None
        for path in possible_paths:
            if os.path.exists(path):
                env_path = path
                self.stdout.write(f"[OK] Archivo .env encontrado en: {path}")
                break
        
        if env_path:
            load_dotenv(env_path, override=True)
        
        # Intenta leer variable de varias formas
        api_key = os.environ.get('openai_apikey')
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            self.stderr.write("[ERROR] OpenAI API key no encontrada.")
            self.stderr.write("   Solucion: Crea archivo openAI.env con:")
            self.stderr.write("   openai_apikey=sk-proj-...")
            return
        
        self.stdout.write(f"[OK] API key cargada correctamente (primeros 20 chars: {api_key[:20]}...)")
        client = OpenAI(api_key=api_key)

        # ✅ Fetch all movies from the database
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in the database")

        def get_embedding(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        # ✅ Iterate through movies and generate embeddings
        count = 0
        for movie in movies:
            try:
                emb = get_embedding(movie.description)
                # ✅ Store embedding as binary in the database
                movie.emb = emb.tobytes()
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"[OK] Embedding stored for: {movie.title}"))
                count += 1
            except Exception as e:
                Dself.stderr.write(f"[ERROR] Failed to generate embedding for {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS(f"[DONE] Finished generating embeddings for {count} movies"))

        # ✅ Show random movie embedding if requested
        if kwargs.get('show_random') or movies.exists():
            self._show_random_embedding(movies)

    def _show_random_embedding(self, movies):
        """Display embeddings of a random movie"""
        if not movies.exists():
            self.stdout.write(self.style.WARNING("No movies found in database"))
            return
        
        # ✅ Get a random movie
        random_movie = random.choice(list(movies))
        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.SUCCESS(f"[RANDOM MOVIE] {random_movie.title}"))
        self.stdout.write("="*80)
        
        self.stdout.write(f"\n[DESCRIPTION] {random_movie.description[:200]}...")
        self.stdout.write(f"[GENRE] {random_movie.genre}")
        self.stdout.write(f"[YEAR] {random_movie.year}")
        
        # ✅ Convert binary embedding back to numpy array
        if random_movie.emb:
            try:
                embedding = np.frombuffer(random_movie.emb, dtype=np.float32)
                self.stdout.write(f"\n[EMBEDDING] Shape: {embedding.shape}")
                self.stdout.write(f"[EMBEDDING] Size: {len(embedding)} dimensions")
                self.stdout.write(f"[EMBEDDING] Mean: {embedding.mean():.6f}")
                self.stdout.write(f"[EMBEDDING] Std Dev: {embedding.std():.6f}")
                self.stdout.write(f"[EMBEDDING] Min: {embedding.min():.6f}")
                self.stdout.write(f"[EMBEDDING] Max: {embedding.max():.6f}")
                
                # ✅ Show first 20 embedding values
                self.stdout.write("\n[VALUES] First 20 Embedding Values:")
                self.stdout.write("-" * 80)
                for i in range(min(20, len(embedding))):
                    self.stdout.write(f"  [{i:4d}]: {embedding[i]:10.6f}")
                self.stdout.write("  ...")
                self.stdout.write("-" * 80)
                
            except Exception as e:
                self.stderr.write(f"[ERROR] Error reading embedding: {e}")
        else:
            self.stdout.write("[ERROR] No embedding found for this movie")

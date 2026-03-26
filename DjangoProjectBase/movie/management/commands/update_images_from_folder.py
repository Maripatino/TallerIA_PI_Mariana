import os
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Update movie images from the media/movie/images/ folder"

    def handle(self, *args, **kwargs):
        # Fetch all movies from the database
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        # Process each movie
        for movie in movies:
            self.stdout.write(f"Processing: {movie.title}")
            try:
                # Construct the image path (assuming images are named like m_{title}.png)
                image_filename = f"m_{movie.title}.png"
                image_path = os.path.join('movie', 'images', image_filename)

                # Update the database with the image path
                movie.image = image_path
                movie.save()

                self.stdout.write(self.style.SUCCESS(f"Updated image for: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {str(e)}")

        self.stdout.write(self.style.SUCCESS("Process finished."))
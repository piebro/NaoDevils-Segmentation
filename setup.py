from setuptools import find_packages, setup


setup(name="robocup_data_loader",
      version="0.1.0",
      description="Data Loader for the nao robocup dataset",
      author="Piet Broemmel",
      author_email='piet.broemmel@gmail.com',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="MIT",
      url="https://github.com/piebro/robocup-dataset-loader",
      packages=find_packages(exclude=["test"]),
      install_requires=[
            "png",
            "itertools",
            "pycocotools",
            "opencv-python",
            "numpy",
            "google.protobuf",
            "imgaug",
            "matplotlib"]
)
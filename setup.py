from setuptools import find_packages, setup


setup(name="robocup_dataset_loader",
      version="0.1.0",
      description="Data Loader for the nao robocup dataset",
      author="Piet Broemmel",
      author_email='piet.broemmel@gmail.com',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="MIT",
      url="https://github.com/piebro/robocup-dataset-loader",
      packages=find_packages(),
      install_requires=[
            "png==0.0.20",
            "itertools==8.2.0",
            "pycocotools==2.0.0",
            "opencv-python==4.1.2.30",
            "numpy==1.17.5",
            "google==2.0.3",
            "imgaug==0.2.9",
            "matplotlib==3.1.3"]
)
from setuptools import setup, find_packages

setup(
   name='iaif',
   version='1.0',
   author='Markus Klar, Samuel Younger',
   author_email='markus.klar@glasgow.ac.uk, Samuel.Younger24@imperial.ac.uk',
   packages=['iaif'],
   package_data={'': ['**']},
   url='https://github.com/youngers2006/iaif.git',
   license='LICENSE',
   python_requires='>=3.11',
   install_requires=[
       "numpy", "optax", "tqdm", "pyyaml", "matplotlib", "pandas", "seaborn", "scikit-learn", "ipykernel", "difai @ git+https://github.com/youngers2006/difai-base.git@main"
   ],
   extras_require={
    "gpu": ["jax[cuda13]"],
    "tpu": ["jax[tpu]"],
    "cpu": ["jax"],
    },
)
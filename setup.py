from setuptools import setup, find_packages

setup(
    name='ml_algo_lib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.0.0',
        'typing>=3.7.4',
    ],
    author='Kevin Yu',
    author_email='kevin@clique.tech', 
    description='A comprehensive machine learning library with parallel processing, callbacks, and custom metrics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kvnyu24/ml_algo_lib',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requires='>=3.7',
)
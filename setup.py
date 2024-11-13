from setuptools import setup, find_packages

setup(
    name='ml_algo_lib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0'
    ],
    author='Kevin Yu',
    author_email='kevin@clique.tech',
    description='A comprehensive machine learning library',
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
    ],
    python_requires='>=3.7',
) 
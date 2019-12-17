import setuptools


setuptools.setup(
    name="nemo_nlp",
    version="0.8.2",
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    description="Collection of Neural Modules for Natural Language Processing",
    long_description_content_type="text/markdown",
    url="https://github.com/nvidia/nemo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License"
    ],
    install_requires=[
        'nemo_toolkit',
        'torchtext',
        'sentencepiece',
        'python-dateutil<2.8.1,>=2.1',
        'boto3',
        'unidecode',
        'pytorch-transformers',
        'matplotlib',
        'youtokentome'
    ]
)

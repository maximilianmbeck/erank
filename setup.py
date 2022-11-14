#!/usr/bin/env python

import setuptools

setuptools.setup(name='erank',
                 version='0.0',
                 description='Research Repository erank',
                 author='Maximilian Beck',
                 author_email='maximilianbeck@live.de',
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence'
                 ],
                 packages=setuptools.find_packages(),
                 python_requires='>=3.8',
                 install_requires=['numpy', 'torch', 'torchmetrics', 'tqdm', 'matplotlib', 'pandas', 'hydra-core', 'wandb', 'plotly', 'scipy', 'seaborn']
                 )

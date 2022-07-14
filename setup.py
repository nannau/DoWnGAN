import sys
from setuptools import setup, find_packages


setup(
    name="DoWnGAN",
    description="DOwnscaling with WassersteiN Generative Adversarial Networks for Super Resolution",
    keywords="AI deep learning generative super resolution downscaling",
    packages=find_packages(),
    version="0.1dev",
    author="Nic Annau",
    author_email="nannau@uvic.ca",
    zip_safe=True,
    scripts=[
        "DoWnGAN/GAN/train.py",
        "DoWnGAN/GAN/dataloader.py",
        "DoWnGAN/GAN/losses.py",
        "DoWnGAN/config/config.py",
        "DoWnGAN/config/hyperparams.py",
        "DoWnGAN/helpers/prep_gan.py",
        "DoWnGAN/helpers/wrf_times.py",
        "DoWnGAN/helpers/gen_experiment_datasets.py",
        "DoWnGAN/networks/critic.py",
        "DoWnGAN/networks/generator.py",
        "DoWnGAN/mlflow_tools/gen_grid_plots.py",
    ],
    install_requires=["numpy", "dask", "torch", "xarray", "sklearn"],
    extras_require = {"dask": "distributed"},
    package_dir={"DoWnGAN": "DoWnGAN"},
    package_data={"DoWnGAN": ["data/*", "DoWnGAN/"]},
    classifiers="""
        Intended Audience :: Science/Research
        License :: GNU General Public License v3 (GPLv3)
        Operating System :: OS Independent
        Hardware :: Requires CUDA GPU
        Programming Language :: Python :: 3.8
        Topic :: Scientific/Engineering
        Topic :: Software Development :: Libraries :: Python Modules""".split(
                "\n"
    ),
)
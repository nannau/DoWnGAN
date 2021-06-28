import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ["-v", "tests"]
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name="DoWnGAN",
    description="Wasserstein Generative Adversarial Networks for Wind Field Super Resolution",
    keywords="AI deep learning generative super resolution downscaling",
    packages=find_packages(),
    version="0.1dev",
    author="Nic Annau",
    author_email="nannau@uvic.ca",
    zip_safe=True,
    scripts=[
        "DoWnGAN/dataloader.py",
        "DoWnGAN/gen_plots.py",
        "DoWnGAN/losses.py",
        "DoWnGAN/prep_gan.py",
        "DoWnGAN/process_data.py",
        "DoWnGAN/training.py",
        "DoWnGAN/utils.py",
        "DoWnGAN/run.py",
        "DoWnGAN/models/critic.py",
        "DoWnGAN/models/generator.py"
    ],
    install_requires=["numpy", "torch", "xarray", "sklearn"],
    tests_require=["pytest"],
    cmdclass={"test": PyTest},
    package_dir={"DoWnGAN": "DoWnGAN"},
    package_data={"DoWnGAN": ["tests/*", "data/*", "DoWnGAN/"]},
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
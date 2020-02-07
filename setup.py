import setuptools

setuptools.setup(
    name='nas',
    version='1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow>=1.12.0',
    ]
)
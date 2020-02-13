import setuptools

setuptools.setup(
    name='nas',
    version='1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        # 'tensorflow==1.13.2',
        'tensorflow-gpu==1.13.2',
    ]
)
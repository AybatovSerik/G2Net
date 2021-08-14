from setuptools import setup, find_packages
import G2Net
print(find_packages(where="G2Net"))

setup(
    name='G2Net',
    version=G2Net.__version__,
    packages=['G2Net'],
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    description='Layers and models for G2Net',
    url='https://github.com/AybatovSerik/G2Net',
    author='Aybatov Serik',
    author_email='aybatov.serik@gmail.com',
)
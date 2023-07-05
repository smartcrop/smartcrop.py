"""
    smartcrop.py
    ~~~~~~~~~~~~

    smartcrop.js implementation in Python

    :license: MIT
"""
from setuptools import setup

setup(
    name='smartcrop',
    entry_points={
        'console_scripts': ['smartcroppy=smartcrop:main']
    },
    include_package_data=True,
    install_requires=['numpy', 'Pillow>=4.0.0', 'PyGObject', 'pytoolbox[imaging]>=14.5.1'],
    py_modules=['smartcrop'],
    zip_safe=False,

    # Meta-data for upload to PyPI
    author='Hideo Hattori',
    author_email='hhatto.jp@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities'
    ],
    description='smartcrop implementation in Python',
    keywords=['image', 'crop', 'PIL', 'Pillow'],
    license='MIT',
    long_description=open('README.rst').read(),
    platforms='any',
    url='https://github.com/smartcrop/smartcrop.py',
    version='0.3.4'
)

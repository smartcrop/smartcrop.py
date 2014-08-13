"""
    smartcrop.py
    ~~~~~~~~~~~~

    smartcrop.js implementation in Python

    :license: MIT
"""

from setuptools import setup

setup(
    name='smartcrop',
    version='0.1',
    description="smartcrop implementation in Python",
    long_description=open('README.rst').read(),
    author="Hideo Hattori",
    author_email="hhatto.jp@gmail.com",
    keywords=("image", "crop", "PIL", "Pillow"),
    url = "https://github.com/hhatto/smartcrop.py",
    include_package_data=True,
    py_modules=['smartcrop'],
    zip_safe=False,
    platforms = 'any',
    install_requires=['Pillow'],
    license='MIT',
    entry_points={
        'console_scripts': ['smartcroppy = smartcrop:main'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Utilities'
    ]
)

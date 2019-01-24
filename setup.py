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
        'console_scripts': ['smartcroppy = smartcrop:main']
    },
    include_package_data=True,
    install_requires=['Pillow>=4.3.*'],
    py_modules=['smartcrop'],
    use_2to3=True,
    use_2to3_exclude_fixers=['lib2to3.fixes.fix_import'],
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
    url='https://github.com/hhatto/smartcrop.py',
    version='0.2'
)

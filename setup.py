# coding=utf-8
# Creation date: 19 окт. 2020
# Creation time: 9:54
# Creator: SteamPeKa

import setuptools

setuptools.setup(
    name="krippendorffs_alpha",
    version="0.2alpha",
    author="SteamPeKa",
    author_email="vladimir.o.balagurov@yandex.ru",
    description="Krippendorff 's alpha-reliability coefficient computation",
    long_description="",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    test_suite="tests",
    zip_safe=False,
    install_requires=["numpy==1.19.2",
                      "pytest>=2.8.5"]
)

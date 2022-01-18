from setuptools import setup

setup(name='pybullet_test',
      description='Pybullet test',
      version='0.1',
      author='Antonis Sidiropoulos',
      author_email='antosidi@ece.auth.gr',
      install_requires=['numpy==1.18', \
                        'sphinx', \
                        # 'sphinxcontrib-bibtex==2.1.4', \
                        # 'sphinx_rtd_theme==0.5.1', \
                        'numpydoc==1.1.0', \
                        'pybullet==3.2.1', \
                        'opencv-python', \
                        'matplotlib==3.0.3', \
                        'scipy', \
                        'open3d==0.8.0', \
                        'PyYAML', \
                        'torch', \
                        'torchvision', \
                        'tensorflow'
      ]
)
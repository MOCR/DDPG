import glob
from distutils.core  import setup, Extension
deps = glob.glob('src/*.c')
packages          = ['mpl_toolkits','mpl_toolkits.natgrid']
package_dirs       = {'':'lib'}
extensions = [Extension("mpl_toolkits.natgrid._natgrid",deps,include_dirs = ['src'],)]
setup(
  name              = "natgrid",
  version           = "0.2.1",
  description       = "Python interface to NCAR natgrid library",
  url               = "http://matplotlib.sourceforge.net/toolkits.html",
  download_url      = "http://sourceforge.net/projects/matplotlib",
  author            = "Jeff Whitaker",
  author_email      = "jeffrey.s.whitaker@noaa.gov",
  platforms         = ["any"],
  license           = "Restricted",
  keywords          = ["python","plotting","plots","graphs","charts","GIS","mapping","map projections","maps"],
  classifiers       = ["Development Status :: 4 - Beta",
			           "Intended Audience :: Science/Research", 
			           "License :: OSI Approved", 
			           "Topic :: Scientific/Engineering :: Visualization",
			           "Topic :: Software Development :: Libraries :: Python Modules",
			           "Operating System :: OS Independent"],
  packages          = packages,
  package_dir       = package_dirs,
  ext_modules       = extensions,
  )

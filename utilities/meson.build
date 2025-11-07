bash_scripts = [
    'activate_installation'
]

install_data(
    bash_scripts
)

py_sources = [
    '__init__.py',
]

py3.install_sources(
    py_sources,
    pure: false,
    subdir: 'utilities',
)


subdir('dbview')
subdir('help')
subdir('setup')

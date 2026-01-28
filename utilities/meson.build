bash_scripts = [
    'activate_installation'
]

install_data(
    bash_scripts,
    install_dir: get_option('bindir'),
)

py_sources = [
    '__init__.py',
]

py3.install_sources(
    py_sources,
    pure: false,
    subdir: 'lute_utilities',
)


subdir('dbview')
subdir('help')
subdir('setup')

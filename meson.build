project(
    'lute',
    'cpp',
    version: '0.1.0',
    default_options: [
        'cpp_std=c++20',
    ],
)
add_project_arguments('-frtti', language: 'cpp')

pymod = import('python')

py3 = pymod.find_installation(pure: false)
py3_dep = py3.dependency()

min_python_ver = '3.8'
py_ver = py3.language_version()
if py_ver.version_compare(f'<@min_python_ver@')
  error(f'Minimum supported Python version is @min_python_ver@, found @py_ver@')
endif

py_sources = [
    'run_task.py',
    'subprocess_task.py',
]

py3.install_sources(
    py_sources,
    pure: false,
)

subdir('extensions')
subdir('lute')
subdir('utilities')
subdir('workflows/maestro')
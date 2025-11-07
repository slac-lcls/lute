pybind11_dep = dependency('pybind11', required: true)
add_project_arguments('-frtti', language: 'cpp')

cmake = import('cmake')
spdlog_opts = cmake.subproject_options()
spdlog_opts.add_cmake_defines(
    {'SPDLOG_BUILD_ALL':'ON', 'CMAKE_POSITION_INDEPENDENT_CODE':'ON'}
)
spdlog = cmake.subproject('spdlog', options: spdlog_opts)

libspdlog = spdlog.get_variable('spdlog')
spdlog_inc = spdlog.get_variable('spdlog_inc')
spdlog_dep = declare_dependency(
    link_with: libspdlog,
    include_directories: spdlog_inc,
)

liblwmhttp = shared_library(
    'lwmhttp',
    ['src/server/handler.cc',
     'src/server/http.cc',
     'src/server/server.cc'],
    dependencies: [spdlog_dep],
    include_directories: spdlog_inc,
    install: true,
    install_dir: py3.get_install_dir() / 'maestro/_maestro',
    version: '1',
    soversion: '1',
    link_args: ['-lpthread'],
)

lwm_http_dep = declare_dependency(
    link_with: [libspdlog, liblwmhttp],
    dependencies: [spdlog_dep],
)

liblwm = shared_library(
    'lwm',
    ['src/lwm/job.cc',
     'src/lwm/launcher.cc',
     'src/lwm/manager.cc',
     'src/parallel/threadpool.cc'],
    install: true,
    install_dir: py3.get_install_dir() / 'maestro/_maestro',
    dependencies: [spdlog_dep, lwm_http_dep],
    include_directories: spdlog_inc,
    version: '1',
    soversion: '1',
    link_args: ['-lpthread'],
)

lwm_dep = declare_dependency(
    link_with: [liblwm, liblwmhttp],
    dependencies: [lwm_http_dep],
)

py_mod = py3.extension_module('_maestro',
    ['src/_maestro/maestro.cc'],
    install: true,
    dependencies : [pybind11_dep, lwm_dep, spdlog_dep],
    subdir: 'maestro/_maestro',
    install_rpath: '$ORIGIN',
)

# Install the Python modules
# Could do this "properly" with subdir and iterative meson.build files but its
# only a handful of modules
py_sources = [
    'src/maestro/__init__.py',
    'src/maestro/launch_maestro.py',
    'src/maestro/parser.py',
]
py3.install_sources(
    py_sources,
    subdir: 'maestro',
)
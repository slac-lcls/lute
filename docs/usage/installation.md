# Installing `LUTE`
## Source Code

The source code for `LUTE` is available on GitHub at [slac-lcls/lute](https://github.com/slac-lcls/lute).

It can therefore be retrieved on the command-line using:
```bash
> git clone git@github.com:slac-lcls/lute
# or git clone https://github.com/slac-lcls/lute.git for https
```

## Building the Package

At the top level of the directory is a script `build.sh` for building the code on S3DF.

You can run it using:
```bash
> ./build.sh # in top level of the repository you just cloned
```

This will create an `install` directory inside the top-level of `LUTE`, and build and install all components there.

The `build.sh` script will create a build environment and cache it in your home directory (under `~/.cache/lute_build_env_XXXX`). The full path is determined from a hash of the installation location. Caching this build environment speeds up subsequent re-builds significantly; however, it can be deleted by passing `-c` to the `build.sh` script when it is run. The full set of options for this script are:

```bash
> ./build.sh -h
build.sh:
    Build an installation of LUTE.

    This build script will create an isolated build environment. It is cached in
    your home directory under ~/.cache/lute_build_env_XXXX where the final portion
    is created from a hash of base installation directory.

    Subsequent runs of the build will not need to re-create the build environment,
    speeding up the process significantly. You can of course delete the build environment
    from the specified folder at any time, and it will be recreated next time the script
    is run. You can pass a parameter to this script to do cleanup as well.

    Options:
        -c|--clean
          Clean up the build environment
        -h|--help
          Display this message.

    Options that apply on subsequent runs of the build script:
        -e|--entry_points
          Re-run the pip install command. This is only needed if pyproject.toml is
          modified.
        -r|--reconfigure
          Re-run the meson setup. This is only required if meson.build files have been
          modified, or meson options/the install prefix have changed since the last
          time it was run.
```

The installation can be "activated" after being built. This will put all the relevant scripts and binaries into your path.

```bash
> source install/bin/activate_installation
```

This activation script is sourced automatically by submission scripts to make the LUTE code available to various job steps.

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

The installation can be "activated" after being built. This will put all the relevant scripts and binaries into your path.

```bash
> source install/bin/activate_installation
```

This activation script is sourced automatically by submission scripts to make the LUTE code available to various job steps.

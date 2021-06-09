# Generatools
Toolbox for text generation

## Install
### Cuda 11.0
See [here](https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal).

Beforehand, you may want to uninstall previous nvidia drivers etc.. See [here](https://stackoverflow.com/a/62276101).

### Pytorch
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Other requirements
```bash
pip install -r requirements.txt
```


### Contributing

#### Conventions
**Linting**

Before commiting, use `black` for code formatting, with line-length set to 79.
```bash
black . -l 79
```

**Printing and logging**

All the priting should happen through a logger. Do not use `print`.

In a module, iniate a minimal logger only, so that the user's logger config won't be
  overriden
```python
import logging
logger = logging.getLogger(__name__)
```

In a script, the following can be used:
```python
import logging
logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
```

**Exceptions**

If you need to raise an error, use `mnemgen.utils.logging.log_and_raise`. It will
both log an error and raise an exception.

**Typing hint**

Following Google style rules [here](https://google.github.io/styleguide/pyguide.html#s2.21-type-annotated-code) and [here](https://google.github.io/styleguide/pyguide.html#s3.19-type-annotations).

This makes the code easy to inspect statically. Especially, the IDE becomes
able to show the definition of any argument, which helps understanding their
manipulation.

**Docstring**

Docstrings follow the [numpy conventions](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy).

Save time using vim-pydocstring.

Docstrings must be thorough for classes and functions used by the end user.
They may be much lighter for private functions, methods and classes.

For each module, put a docstring in the `__init__.py` file.

**Testing**
Tests should be marked:
* Slow tests: use decorator `@pytest.mark.slow`
* Unit tests: use decorator `@pytest.mark.unit`

Distinguising unit tests (implementation) from functional tests. If a unit
tests fails but the implementation test don't, this may simply be related to an
implementation change.

Distinguishing slow tests from others allow running fast tests all the time, and slow tests less often.

You may use both pytest and unittest. When running tests:
* Faster tests only: `py.test`
* Including slow tests: `py.test --runslow`

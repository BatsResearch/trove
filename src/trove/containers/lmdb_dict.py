"""Memory mapped dict based using LMDB.

Adopted from here: https://github.com/Dobatymo/lmdb-python-dbm/tree/master
"""

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

import lmdb

from .. import file_utils


class LMDBDict:
    """A memory mapped variant of python dictionary with limited functionalities.

    You can use this class to decrease the memory footprint of large python dictionaries.


    ..  warning::

        This is WIP. Use this class with extra causion to avoid logical bugs.

    If the following conditions are not met, it is not guaranteed that this class functions as expected and correctly:

        - Only one worker at a time (across processes, threads, etc.) should write to this container.
        - Do not update or overwrite the database file after its creation. I.e., no write operations after you created the database file.
        - Make sure you instantiate reader instances of this class after its creation is completed.
          Readers that are created before the write operation ends might represent an inconsistent view of the data.

    You can follow these requirements by simply doing the following in your code:

    .. code-block:: python

        with file_utils.easyfilelock(db_file):
            if not Path(db_file).exists():
                LMDBDict.create_db(data=dict(...), db_file=db_file)
        mm_dict = LMDBDict.load_db(db_file=db_file, ...)
        print(mm_dict[b'some_key'])
        mm_dict.close()

    You can also use this class as a context manager to properly manage opening
    and closing LMDB environments.

    .. code-block:: python

        with LMDBDict.load_db(...) as db:
            print(db[b'some_key'])

    ..  note::

        from `LMDB docs <http://www.lmdb.tech/doc/starting.html>`_:

        LMDB uses POSIX locks on files, and these locks have issues if
        one process opens a file multiple times. Because of this, do
        not mdb_env_open() a file multiple times from a single process.
        Instead, share the LMDB environment that has opened the file
        across all threads. Otherwise, if a single process opens the
        same environment multiple times, closing it once will remove
        all the locks held on it, and the other instances will be
        vulnerable to corruption from other processes.

    To make sure, we only create one lmdb environment for each file, we save the path to
    the DB file and its corresponding environment in ``lmdb_envs`` class attribute whenever we
    create a new environment. If the user calls ``LMDBDict.load_db()`` for the same file again,
    we reuse the saved environment.
    """

    lmdb_envs = dict()

    def __init__(
        self,
        env: lmdb.Environment,
        encode_key: Optional[Callable[[Any], bytes]] = None,
        decode_key: Optional[Callable[[bytes], Any]] = None,
        encode_value: Optional[Callable[[Any], bytes]] = None,
        decode_value: Optional[Callable[[bytes], Any]] = None,
    ) -> None:
        """Initialize a read-only memory mapped dict with data saved in the given lmdb environment.

        The remaining arguments are the same as arguments to :py:meth:`trove.containers.lmdb_dict.LMDBDict.load_db`.

        Args:
            env: the LMDB environment that saves the underlying data.
        """
        self.env = env
        self.encode_key = encode_key if encode_key is not None else lambda x: x
        self.decode_key = decode_key if decode_key is not None else lambda x: x
        self.encode_value = encode_value if encode_value is not None else lambda x: x
        self.decode_value = decode_value if decode_value is not None else lambda x: x

    @classmethod
    def load_db(
        cls,
        db_file: os.PathLike,
        db_hash: Optional[str] = None,
        encode_key: Optional[Callable[[Any], bytes]] = None,
        decode_key: Optional[Callable[[bytes], Any]] = None,
        encode_value: Optional[Callable[[Any], bytes]] = None,
        decode_value: Optional[Callable[[bytes], Any]] = None,
    ):
        """Load a memory mapped dict from disk.

        Args:
            db_file: path to lmdb database file that holds the data.
            db_hash: If given, it checks if hash of the given ``db_file`` is equal to ``db_hash``.
                Since we are not properly closing the readonly environments on exit, this optional hash
                comparison ensures that DB files are not corrupted in previous runs.
            encode_key: callable with one positional argument that takes in a dictionary key and encodes into bytestring.
                If not provided, keys must be of type ``bytes`` already.
            decode_key: callable with one positional argument that takes in a bytestring and returns the decoded dict key.
                If not provided, this class returns the saved dict key which is of type ``bytes``.
            encode_value: callable with one positional argument that takes in a dict value and encodes into bytestring.
                If not provided, values must be of type ``bytes`` already.
            decode_value: callable with one positional argument that takes in a bytestring and returns the decoded dict value.
                If not provided, this class returns the saved dict value which is of type ``bytes``.

        Returns:
            an instance of LMDBDict populated with the data from the given db file.
        """
        if not Path(db_file).exists():
            msg = f"DB file does not exists:\n{db_file}"
            raise RuntimeError(msg)

        _real_filepath = file_utils.realpath(db_file)
        if _real_filepath in cls.lmdb_envs:
            # We've already created/opened an environment for this db file in this process.
            # Reuse the previous environment instance. Accoding to LMDB docs, we should create one
            # environment for each db file in each process.
            env = cls.lmdb_envs[_real_filepath]["env"]
            # Keep track of the number of instances that use this environment instance.
            # Useful for a proper cleanup-on-exit procedure in the future
            cls.lmdb_envs[_real_filepath]["num_refs"] += 1
        else:
            if db_hash is not None:
                # If a hash is provided, make sure the db file is not corrupted.
                file_hash = file_utils.hash_file_bytes(path=_real_filepath)
                if file_hash.strip() != db_hash.strip():
                    msg = f"DB file content has changed after it has been created. DB file: '{db_file}'"
                    raise RuntimeError(msg)

            env = lmdb.Environment(
                path=_real_filepath,
                map_size=100 * 2**30,
                subdir=False,
                readonly=True,
                readahead=False,
                max_dbs=0,
            )
            # save a pointer to this environment to potentially create other
            # LMDBDict instances with the same DB file in the future
            cls.lmdb_envs[_real_filepath] = {"env": env, "num_refs": 1}
        obj = cls(
            env=env,
            encode_key=encode_key,
            decode_key=decode_key,
            encode_value=encode_value,
            decode_value=decode_value,
        )
        return obj

    @classmethod
    def create_db(
        cls,
        data: Iterable[Tuple[Any, Any]],
        db_file: os.PathLike,
        encode_key: Optional[Callable[[Any], bytes]] = None,
        encode_value: Optional[Callable[[Any], bytes]] = None,
    ) -> str:
        """Create an LMDB database from the given data.

        To use the created database, you should load it into an instance of LMDBDict:

        .. code-block:: python

            LMDBDict.create_db(data=[(b'a', b'val'), (b'b', b'v2')], db_file='./somefile.db')
            mm_dict = LMDBDict.load_db(db_file='./somefile.db')
            assert mm_dict[b'a'] == b'val'

        Args:
            data: an iterable of (key, value) 2-tuples. E.g., ``[(b'a', b'aa'), (b'c', b'cc')]``.
                We do not verify the data type of keys and values. You must make sure to provide
                keys and values of correct data type (i.e., ``bytes```).
            db_file: Path to the destination DB file.
            encode_key: callable with one positional argument that takes in a dict key and encodes into bytestring.
                If not provided, keys must be of type ``bytes`` already.
            encode_value: callable with one positional argument that takes in a dict value and encodes into bytestring.
                If not provided, values must be of type ``bytes`` already.

        Returns:
            hash of the created database file.
        """
        # just to avoid extra if statements in this function
        if encode_key is None:
            encode_key = lambda x: x
        if encode_value is None:
            encode_value = lambda x: x

        Path(db_file).parent.mkdir(exist_ok=True, parents=True)
        _real_fp = file_utils.realpath(db_file)
        with file_utils.easyfilelock(str(_real_fp) + ".is_writing"):
            # Weak file locking. This only ensures no instance of this function is writing to this file.
            # Other functions can still modify this file. But, that is ok, we ask the user
            # to make sure that no other processes is writing to the db file when calling this function.
            # See class docstrings for more info.
            with file_utils.atomic_write(file=_real_fp, root="parent") as tfp:
                with lmdb.Environment(
                    path=tfp,
                    map_size=100 * 2**30,
                    subdir=False,
                    readonly=False,
                    mode=493,
                    readahead=False,
                    max_dbs=0,
                ) as env:
                    with env.begin(write=True) as txn:
                        with txn.cursor() as curs:
                            curs.putmulti(
                                (encode_key(k), encode_value(v)) for k, v in data
                            )
                        env.sync()
            db_hash = file_utils.hash_file_bytes(path=_real_fp)
        return db_hash

    def close(self) -> None:
        """Close LMDB environment.

        It reduces the number of references to the current environment by one (i.e., soft close).
        If there are no remaining references to this environment, it closes the environment.
        """
        db_path = self.env.path()
        self.__class__.lmdb_envs[db_path]["num_refs"] -= 1
        if self.__class__.lmdb_envs[db_path]["num_refs"] < 1:
            self.__class__.lmdb_envs[db_path]["env"].close()

    def __enter__(self):
        """Act as a context manager."""
        return self

    def __exit__(self, *args, **kwargs):
        """Cleanups before exiting the context."""
        self.close()

    def get(self, key: Any, default: Optional[Any]) -> Any:
        """Similar to python's ``dict.get`` method.

        But key must be of type bytes.
        """
        with self.env.begin() as txn:
            value = txn.get(self.encode_key(key))
        if value is None:
            return default
        else:
            return self.decode_value(value)

    def keys(self):
        """Similar to python's ``dict.keys``."""
        with self.env.begin() as txn:
            with txn.cursor() as curs:
                for k in curs.iternext(keys=True, values=False):
                    yield self.decode_key(k)

    def values(self):
        """Similar to python's ``dict.values``."""
        with self.env.begin() as txn:
            with txn.cursor() as curs:
                for v in curs.iternext(keys=False, values=True):
                    yield self.decode_value(v)

    def items(self):
        """Similar to python's ``dict.items``."""
        with self.env.begin() as txn:
            with txn.cursor() as curs:
                for k, v in curs.iternext(keys=True, values=True):
                    yield (self.decode_key(k), self.decode_value(v))

    def __len__(self) -> int:
        """Similar to python's ``len(dict)``."""
        with self.env.begin() as txn:
            num_entries = txn.stat()["entries"]
        return num_entries

    def __contains__(self, key: Any) -> bool:
        """Support ``in`` operator like ``key in dict``."""
        with self.env.begin() as txn:
            value = txn.get(self.encode_key(key))
        return value is not None

    def __iter__(self):
        """Support iterating directly over LMDBDict instances similar to python's dictionary."""
        return self.keys()

    def __getitem__(self, key: Any) -> Any:
        """support indexing operator."""
        with self.env.begin() as txn:
            value = txn.get(self.encode_key(key))
        if value is None:
            raise KeyError(key)
        return self.decode_value(value)

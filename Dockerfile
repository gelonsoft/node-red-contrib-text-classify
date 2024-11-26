FROM node:22-bookworm
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LANG C.UTF-8

# runtime dependencies
RUN set -eux; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
                libbluetooth-dev \
                tk-dev \
                uuid-dev \
        ; \
        rm -rf /var/lib/apt/lists/*

ENV GPG_KEY E3FF2839C048B25C084DEBE9B26995E310250568
ENV PYTHON_VERSION 3.9.20

RUN set -eux; \
        \
        wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
        wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"; \
        GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; \
        gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$GPG_KEY"; \
        gpg --batch --verify python.tar.xz.asc python.tar.xz; \
        gpgconf --kill all; \
        rm -rf "$GNUPGHOME" python.tar.xz.asc; \
        mkdir -p /usr/src/python; \
        tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz; \
        rm python.tar.xz; \
        \
        cd /usr/src/python; \
        gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
        ./configure \
                --build="$gnuArch" \
                --enable-loadable-sqlite-extensions \
                --enable-optimizations \
                --enable-option-checking=fatal \
                --enable-shared \
                --with-system-expat \
                --with-ensurepip \
        ; \
        nproc="$(nproc)"; \
        EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
        LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
        make -j "$nproc" \
                "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
                "LDFLAGS=${LDFLAGS:-}" \
                "PROFILE_TASK=${PROFILE_TASK:-}" \
        ; \
# https://github.com/docker-library/python/issues/784
# prevent accidental usage of a system installed libpython of the same version
        rm python; \
        make -j "$nproc" \
                "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
                "LDFLAGS=${LDFLAGS:--Wl},-rpath='\$\$ORIGIN/../lib'" \
                "PROFILE_TASK=${PROFILE_TASK:-}" \
                python \
        ; \
        make install; \
        \
# enable GDB to load debugging data: https://github.com/docker-library/python/pull/701
        bin="$(readlink -ve /usr/local/bin/python3)"; \
        dir="$(dirname "$bin")"; \
        mkdir -p "/usr/share/gdb/auto-load/$dir"; \
        cp -vL Tools/gdb/libpython.py "/usr/share/gdb/auto-load/$bin-gdb.py"; \
        \
        cd /; \
        rm -rf /usr/src/python; \
        \
        find /usr/local -depth \
                \( \
                        \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
                        -o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
                \) -exec rm -rf '{}' + \
        ; \
        \
        ldconfig; \
        \
        export PYTHONDONTWRITEBYTECODE=1; \
        python3 --version; \
        \
        pip3 install \
                --disable-pip-version-check \
                --no-cache-dir \
                --no-compile \
                'setuptools==58.1.0' \
                wheel \
        ; \
        pip3 --version

# make some useful symlinks that are expected to exist ("/usr/local/bin/python" and friends)
RUN set -eux; \
        for src in idle3 pip3 pydoc3 python3 python3-config; do \
                dst="$(echo "$src" | tr -d 3)"; \
                [ -s "/usr/local/bin/$src" ]; \
                [ ! -e "/usr/local/bin/$dst" ]; \
                ln -svT "$src" "/usr/local/bin/$dst"; \
        done
RUN set -eux; apt-get update; apt-get install -y libaio1
EXPOSE 1880
RUN pip3 install optuna scikit-learn evaluate transformers[torch] datasets nlp pandas nltk
RUN python -c "import nltk ; nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('wordnet')"
RUN mkdir -p /opt/data
RUN set -eux; apt-get install -y vim
RUN npm install -g node-red-contrib-text-classify
RUN echo "#\!/bin/sh\nnpm i -g node-red-contrib-text-classify\nnpx node-red $*" > /opt/run.sh
CMD ["sh","/opt/run.sh"]
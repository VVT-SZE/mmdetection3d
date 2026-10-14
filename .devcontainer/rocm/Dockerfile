FROM rocm/dev-ubuntu-22.04:5.6@sha256:eed7451c3e5dea51bee5d7fde69274663acf9984e3adad30ebf4e81f29d21d29

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/mmdet3d-venv \
    ROCM_HOME=/opt/rocm \
    PYTORCH_ROCM_ARCH=gfx1030 \
    MAX_JOBS=4 \
    PATH=/opt/mmdet3d-venv/bin:/opt/rocm/bin:${PATH}

RUN apt-get -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates curl ffmpeg git g++ libgl1 libglib2.0-0 \
        libsm6 libxext6 libxrender1 openssh-client sudo tar xvfb \
        python3.10-dev python3.10-venv \
        hipblas-dev hipsparse-dev rocrand-dev \
        rocprim-dev rocthrust-dev hipcub-dev \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && python3.10 -m venv "${VIRTUAL_ENV}"

COPY install-stack.sh clear_execstack.py requirements.txt /opt/mmdet3d-rocm/
RUN bash /opt/mmdet3d-rocm/install-stack.sh \
    && chown -R "${USERNAME}:${USERNAME}" "${VIRTUAL_ENV}"

USER ${USERNAME}
WORKDIR /workspace/mmdetection3d

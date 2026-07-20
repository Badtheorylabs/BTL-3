# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6 AS build

ARG CUDA_ARCHITECTURES
RUN test -n "${CUDA_ARCHITECTURES}"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential ca-certificates cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_UI=OFF \
    && cmake --build build --target llama-server llama-cli --parallel

RUN mkdir -p /stage \
    && cp -L build/bin/llama-server build/bin/llama-cli /stage/ \
    && find build/bin -maxdepth 1 -type f -name '*.so*' \
        -exec cp -L '{}' /stage/ ';' \
    && cp LICENSE /stage/LICENSE.llama.cpp \
    && if [ -f /usr/local/cuda/EULA.txt ]; then \
        cp /usr/local/cuda/EULA.txt /stage/LICENSE.NVIDIA-CUDA; \
       fi \
    && for file in /stage/llama-server /stage/llama-cli /stage/*.so*; do \
         [ -e "$file" ] || continue; \
         ldd "$file" 2>/dev/null | awk ' \
           $2 == "=>" && substr($3, 1, 1) == "/" { print $3 } \
           substr($1, 1, 1) == "/" { print $1 }'; \
       done | sort -u | while read -r library; do \
         case "$library" in \
           /lib/*|/usr/lib/*|*libcuda.so*) ;; \
           *) cp -L "$library" "/stage/$(basename "$library")" ;; \
         esac; \
       done

FROM scratch AS export
COPY --from=build /stage/ /

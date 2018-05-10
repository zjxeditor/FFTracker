#!/bin/sh

SRC_DIR="/Users/jxzhang/Downloads/fftw-3.3.7"
cd $SRC_DIR

# Build for arm
# INSTALL_DIR="/Users/jxzhang/Desktop/fftw3-android/arm"
# mkdir -p $INSTALL_DIR

# target_host=arm-linux-androideabi
# export PATH=/Users/jxzhang/Library/Android/sdk/ndk-bundle/standalone-toolchains/arm/bin:$PATH
# export AR=$target_host-ar
# export AS=$target_host-clang
# export CC=$target_host-clang
# export CXX=$target_host-clang++
# export LD=$target_host-ld
# export STRIP=$target_host-strip
# export CFLAGS="-fPIE -fPIC -march=armv7-a -mthumb"
# export LDFLAGS="-pie -march=armv7-a -Wl,--fix-cortex-a8"

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
# 	--enable-threads \
#     --disable-fortran
# make
# make install

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
#     --enable-long-double \
# 	--enable-threads \
#     --disable-fortran
# make
# make install

# make distclean
# export CFLAGS="-fPIE -fPIC -march=armv7-a -mthumb -mfpu=neon"
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
# 	--enable-float \
# 	--enable-threads \
# 	--enable-neon \
#     --disable-fortran
# make
# make install

# Build for arm64

# INSTALL_DIR="/Users/jxzhang/Desktop/fftw3-android/arm64"
# mkdir -p $INSTALL_DIR

# target_host=aarch64-linux-android
# export PATH=/Users/jxzhang/Library/Android/sdk/ndk-bundle/standalone-toolchains/arm64/bin:$PATH
# export AR=$target_host-ar
# export AS=$target_host-clang
# export CC=$target_host-clang
# export CXX=$target_host-clang++
# export LD=$target_host-ld
# export STRIP=$target_host-strip
# export CFLAGS="-fPIE -fPIC"
# export LDFLAGS="-pie"

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
# 	--enable-threads \
#     --enable-neon \
#     --disable-fortran
# make
# make install

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
#     --enable-long-double \
# 	--enable-threads \
#     --disable-fortran
# make
# make install

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
# 	--enable-float \
# 	--enable-threads \
# 	--enable-neon \
#     --disable-fortran
# make
# make install


# Build for x86

# INSTALL_DIR="/Users/jxzhang/Desktop/fftw3-android/x86"
# mkdir -p $INSTALL_DIR

# target_host=i686-linux-android
# export PATH=/Users/jxzhang/Library/Android/sdk/ndk-bundle/standalone-toolchains/x86/bin:$PATH
# export AR=$target_host-ar
# export AS=$target_host-clang
# export CC=$target_host-clang
# export CXX=$target_host-clang++
# export LD=$target_host-ld
# export STRIP=$target_host-strip
# export CFLAGS="-fPIE -fPIC"
# export LDFLAGS="-pie"

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
# 	--enable-threads \
# 	--enable-sse2 \
#     --disable-fortran
# make
# make install

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
# 	--enable-float \
# 	--enable-threads \
# 	--enable-sse2 \
#     --disable-fortran
# make
# make install

# make distclean
# ./configure LIBS="-latomic" \
#     --host=$target_host \
#     --prefix=$INSTALL_DIR \
#     --enable-shared \
#     --enable-long-double \
# 	--enable-threads \
#     --disable-fortran
# make
# make install


# Build for x86_64

INSTALL_DIR="/Users/jxzhang/Desktop/fftw3-android/x86_64"
mkdir -p $INSTALL_DIR

target_host=x86_64-linux-android
export PATH=/Users/jxzhang/Library/Android/sdk/ndk-bundle/standalone-toolchains/x86_64/bin:$PATH
export AR=$target_host-ar
export AS=$target_host-clang
export CC=$target_host-clang
export CXX=$target_host-clang++
export LD=$target_host-ld
export STRIP=$target_host-strip
export CFLAGS="-fPIE -fPIC"
export LDFLAGS="-pie"

make distclean
./configure LIBS="-latomic" \
    --host=$target_host \
    --prefix=$INSTALL_DIR \
    --enable-shared \
	--enable-threads \
	--enable-sse2 \
    --disable-fortran
make
make install

make distclean
./configure LIBS="-latomic" \
    --host=$target_host \
    --prefix=$INSTALL_DIR \
    --enable-shared \
	--enable-float \
	--enable-threads \
	--enable-sse2 \
    --disable-fortran
make
make install

make distclean
./configure LIBS="-latomic" \
    --host=$target_host \
    --prefix=$INSTALL_DIR \
    --enable-shared \
    --enable-long-double \
	--enable-threads \
    --disable-fortran
make
make install



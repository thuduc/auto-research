cat part_* > vcpkg-export-20260803-072137.zip
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$HOME/offline_libs/scripts/buildsystems/vcpkg.cmake"

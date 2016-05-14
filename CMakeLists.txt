cmake_minimum_required(VERSION 2.8)
project( HazeRemove )
find_package(/usr/local/share/OpenCV/OpenCV )
add_executable( HazeRemove HazeRemove.cpp )
target_link_libraries( HazeRemove ${OpenCV_LIBS} )


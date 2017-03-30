cmake_minimum_required(VERSION 2.8.10)
project(FeRe)

SET(FeRe_VERSION 1) 
SET(FeRe_MINOR_VERSION 0)

FIND_PACKAGE(STATISMO REQUIRED)
FIND_PACKAGE(ITK REQUIRED)
include (${ITK_USE_FILE})

FIND_PACKAGE(VTK REQUIRED)
include (${VTK_USE_FILE})


include_directories(${STATISMO_INCLUDE_DIRS})
link_directories( ${ITK_LIBRARY_DIRS} ${statismo_LIBRARY_DIRS})
link_directories( ${VTK_LIBRARY_DIRS} ${statismo_LIBRARY_DIRS})

add_executable(FeRe femurReconstruction.cxx)


target_link_libraries (FeRe ${ITK_LIBRARIES} ${VTK_LIBRARIES} ${statismo_LIBRARIES})


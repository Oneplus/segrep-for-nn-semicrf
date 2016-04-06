INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(sandbox main.cc)
target_link_libraries(sandbox cnn)

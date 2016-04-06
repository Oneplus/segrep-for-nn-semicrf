INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(semi_crf main.cc semi_crf.cc corpus.cc ../logging.cc ../training_utils.cc ../utils.cc ../layer.cc ../semi_crf_layer.cc)
target_link_libraries(semi_crf cnn ${Boost_LIBRARIES})

INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(crf main.cc crf.cc ../layer.cc ../logging.cc ../training_utils.cc ../utils.cc ../labeler/corpus.cc)
target_link_libraries(crf cnn ${Boost_LIBRARIES})

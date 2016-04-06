INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(labeler
    main.cc model.cc corpus.cc ../logging.cc ../layer.cc ../utils.cc ../training_utils.cc)
target_link_libraries(labeler cnn ${Boost_LIBRARIES})

set(TEST_SOURCES 3D-test)

FOREACH (test_src ${TEST_SOURCES})
    add_executable(${test_src} ${test_src}.cpp)
    target_link_libraries(${test_src} ${THIRD_PARTY_LIBS} ${PROJECT_NAME})
    add_test(${test_src} ${test_src})
ENDFOREACH (test_src)

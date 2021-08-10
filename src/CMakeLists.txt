add_library(${PROJECT_NAME} SHARED
	ReadImages.cpp
	yolo.cpp
)

target_link_libraries(${PROJECT_NAME}
        ${THIRD_PARTY_LIBS})

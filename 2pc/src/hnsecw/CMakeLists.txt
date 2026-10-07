add_executable(hnsecw_cli
    hnsecw_cli.cpp
    hnsecw_single_b2y.cpp
    hnsecw_multi_b2y.cpp
    hnsecw_batch_b2y.cpp
)

target_link_libraries(hnsecw_cli
    ABY::aby
    ENCRYPTO_utils::encrypto_utils
)

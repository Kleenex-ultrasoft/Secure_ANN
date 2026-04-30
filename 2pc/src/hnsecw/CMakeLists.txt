add_executable(hnsecw_cli
    hnsecw_cli.cpp
    hnsecw_single_b2y.cpp
    hnsecw_single_b2a.cpp
    hnsecw_multi_b2y_dyn.cpp
    hnsecw_batch_b2y.cpp
)

target_link_libraries(hnsecw_cli
    ABY::aby
    ENCRYPTO_utils::encrypto_utils
)

add_executable(hnsecw_reshuffle_2pc
    hnsecw_reshuffle_2pc.cpp
)

target_link_libraries(hnsecw_reshuffle_2pc
    ABY::aby
    ENCRYPTO_utils::encrypto_utils
)

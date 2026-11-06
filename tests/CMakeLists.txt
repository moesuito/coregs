# CoreGS test fixtures. Every fixture is self-contained: synthetic GS streams and CPU reference
# vectors only, no game data.
#
# GPU fixtures need a Vulkan 1.2 device; without one they exit 77 and CTest reports Skipped,
# never a pass.

set(COREGS_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(COREGS_SRC "${PROJECT_SOURCE_DIR}/src")

# ---- CPU fixtures -----------------------------------------------------------------------------
add_executable(coregs_memory_tests "${COREGS_SRC}/memory/hw_memory_tests.cpp")
target_link_libraries(coregs_memory_tests PRIVATE coregs_core)
foreach(t IN ITEMS hw_swizzle_matches_donor hw_roundtrip_psms hw_psm_layout hw_v01_3_t8_lanes
        hw_v01_4_z_unaligned hw_reinterp_maps hw_format_predicates hw_transfers hw_rect_matches_pixels)
    add_test(NAME memory_${t} COMMAND coregs_memory_tests ${t})
endforeach()

add_executable(coregs_cache_tests "${COREGS_SRC}/cache/hw_cache_tests.cpp")
target_link_libraries(coregs_cache_tests PRIVATE coregs_core)
foreach(t IN ITEMS hw_cache_geometry hw_clamp_regions hw_dirty_merge hw_ageing hw_cache_operations
        hw_v10_alpha_writeback hw_cache_sources_display_moves hw_cache_budget hw_cache_authority_fuzz)
    add_test(NAME cache_${t} COMMAND coregs_cache_tests ${t})
endforeach()

add_executable(coregs_decide_tests "${COREGS_SRC}/decide/hw_gs_decide_fixture.cpp")
target_link_libraries(coregs_decide_tests PRIVATE coregs_core)
foreach(t IN ITEMS hw_draw_rect_table hw_coverage_1x hw_blend_classes hw_blend_vectors hw_blend_levels
        hw_blend_simplify hw_colour_wrap hw_channel_writes hw_atest_fold hw_atest_methods
        hw_date_methods hw_date_vectors hw_barrier_rules hw_overlap_groups hw_vertex_trace
        hw_primitive_expansion hw_scanmsk hw_texture_shuffle hw_channel_shuffle hw_clear_vectors
        hw_pipeline_keys hw_draw_plan_smoke)
    add_test(NAME decide_${t} COMMAND coregs_decide_tests ${t})
endforeach()

add_executable(coregs_lod_tests "${COREGS_SRC}/lod/hw_gs_lod_fixture.cpp")
target_link_libraries(coregs_lod_tests PRIVATE coregs_core)
foreach(t IN ITEMS hw_tex1_decode hw_lod_formula hw_mip_base hw_mip_nearest hw_mip_filter
        hw_miptbp_layout hw_mip_sizes hw_mtba hw_mtba_rules hw_mip_scaling)
    add_test(NAME lod_${t} COMMAND coregs_lod_tests ${t})
endforeach()

add_executable(coregs_pixel_tests "${COREGS_SRC}/pixel/hw_gs_pixel_fixture.cpp")
target_link_libraries(coregs_pixel_tests PRIVATE coregs_core)
foreach(t IN ITEMS hw_tfx_vectors hw_tfx_sweep hw_texa_vectors hw_texa_sweep hw_fog_vectors hw_fog_sweep
        hw_atest_fold hw_atest_vectors hw_blend_vectors hw_blend_modes hw_blend_sweep
        hw_colclamp_vectors hw_dither_vectors hw_frame_convert hw_fbmask_merge hw_date_vectors
        hw_depth_vectors hw_wrap_vectors hw_pipeline_vectors)
    add_test(NAME pixel_${t} COMMAND coregs_pixel_tests ${t})
endforeach()

add_executable(coregs_display_tests "${COREGS_SRC}/display/hw_gs_display_fixture.cpp")
target_link_libraries(coregs_display_tests PRIVATE coregs_core)
foreach(t IN ITEMS hw_video_mode hw_pcrtc_rects hw_pcrtc_offsets hw_merge_plan hw_merge_compose
        hw_interlace_phase hw_deint_weave_blend hw_deint_bob hw_deint_adaptive
        hw_auto_deinterlace hw_shadeboost hw_scanlines hw_present_rect hw_display_options)
    add_test(NAME display_${t} COMMAND coregs_display_tests ${t})
endforeach()

add_executable(coregs_frontend_tests "${COREGS_SRC}/frontend/tests/hw_m3_tests.cpp")
target_link_libraries(coregs_frontend_tests PRIVATE coregs_core)
foreach(t IN ITEMS hw_gif_split hw_gif_ad_xyz hw_flush_rules hw_autoflush hw_vertex_trace hw_st_rounding
        hw_clut_load hw_clut_snapshot hw_clut_copy_free hw_block_ranges_overlap hw_front_end_fast_paths
        hw_transfers hw_counters hw_palette_segments hw_front_end_drop_on_arrival)
    add_test(NAME frontend_${t} COMMAND coregs_frontend_tests ${t})
endforeach()

add_executable(coregs_cpuref_tests "${COREGS_SRC}/cpuref/tests/cpuref_tests.cpp")
target_link_libraries(coregs_cpuref_tests PRIVATE coregs_core)
foreach(t IN ITEMS cpuref_coverage_vectors cpuref_line_pixels cpuref_triangle_shared_edge cpuref_flat_provoking
        cpuref_gouraud cpuref_constant_attributes cpuref_sprite_rectangle cpuref_scissor cpuref_depth
        cpuref_texture_nearest_decal cpuref_clut_indexed cpuref_blend_sweep cpuref_in07_aem
        cpuref_capture_replay cpuref_ppm cpuref_display_frame)
    add_test(NAME cpuref_${t} COMMAND coregs_cpuref_tests ${t})
endforeach()

add_executable(coregs_cpuref_v2_tests "${COREGS_SRC}/cpuref/tests/cpuref_replay_v2_tests.cpp")
target_link_libraries(coregs_cpuref_v2_tests PRIVATE coregs_core)
foreach(t IN ITEMS parity resets fifo malformed)
    add_test(NAME cpuref_capture_v2_${t} COMMAND coregs_cpuref_v2_tests ${t})
endforeach()

add_executable(coregs_cpuref_dump_tests "${COREGS_SRC}/cpuref/tests/cpuref_vram_dump_tests.cpp")
target_link_libraries(coregs_cpuref_dump_tests PRIVATE coregs_core)
foreach(t IN ITEMS selection formats offsets line_double independence metadata)
    add_test(NAME cpuref_vram_dump_${t} COMMAND coregs_cpuref_dump_tests ${t})
endforeach()

add_executable(coregs_replace_tests "${COREGS_SRC}/replace/tests/hw_gs_replace_tests.cpp"
    "${COREGS_SRC}/replace/tests/reference_gs_memory.cpp")
target_link_libraries(coregs_replace_tests PRIVATE coregs_core)
target_include_directories(coregs_replace_tests PRIVATE "${COREGS_SRC}/replace/tests")
add_test(NAME replace_all COMMAND coregs_replace_tests)

add_executable(coregs_core2_upload_tests "${COREGS_SRC}/core2/core2_upload_fixture.cpp")
target_link_libraries(coregs_core2_upload_tests PRIVATE coregs_core)
add_test(NAME core2_upload_swizzle COMMAND coregs_core2_upload_tests)

add_executable(coregs_core2_depth_alias_tests "${COREGS_SRC}/core2/core2_depth_alias_fixture.cpp")
target_link_libraries(coregs_core2_depth_alias_tests PRIVATE coregs_core)
add_test(NAME core2_depth_alias COMMAND coregs_core2_depth_alias_tests)

add_executable(coregs_options_tests "${COREGS_SRC}/hw_options_fixture.cpp")
target_link_libraries(coregs_options_tests PRIVATE coregs_core)
add_test(NAME options COMMAND coregs_options_tests)

add_executable(coregs_device_profile_tests "${COREGS_SRC}/hw_device_profile_fixture.cpp")
target_link_libraries(coregs_device_profile_tests PRIVATE coregs_core)
add_test(NAME device_profile COMMAND coregs_device_profile_tests)

add_executable(coregs_present_latency_tests "${COREGS_SRC}/hw_present_latency_fixture.cpp")
target_link_libraries(coregs_present_latency_tests PRIVATE coregs_core)
add_test(NAME present_latency COMMAND coregs_present_latency_tests)

add_executable(coregs_present_spacing_tests "${COREGS_SRC}/hw_present_spacing_fixture.cpp")
target_link_libraries(coregs_present_spacing_tests PRIVATE coregs_core)
add_test(NAME present_spacing COMMAND coregs_present_spacing_tests)

add_executable(coregs_present_transform_tests "${COREGS_SRC}/hw_present_transform_fixture.cpp")
target_link_libraries(coregs_present_transform_tests PRIVATE coregs_core)
add_test(NAME present_transform COMMAND coregs_present_transform_tests)

add_executable(coregs_present_rect_tests "${COREGS_TESTS_DIR}/present_rect_tests.cpp")
target_link_libraries(coregs_present_rect_tests PRIVATE coregs_core)
add_test(NAME present_rect COMMAND coregs_present_rect_tests)

add_executable(coregs_queue_lock_tests "${COREGS_SRC}/hw_queue_lock_fixture.cpp")
target_link_libraries(coregs_queue_lock_tests PRIVATE coregs_core)
add_test(NAME queue_lock COMMAND coregs_queue_lock_tests)
set_tests_properties(queue_lock PROPERTIES RUN_SERIAL TRUE TIMEOUT 120
    PASS_REGULAR_EXPRESSION "hw_queue_lock: all checks passed")

# ---- GPU fixtures -----------------------------------------------------------------------------
add_executable(coregs_pixel_gpu_tests "${COREGS_SRC}/shaders/hw_pixel_gpu_fixture.cpp")
target_link_libraries(coregs_pixel_gpu_tests PRIVATE coregs_shaders coregs_core coregs_vulkan)
foreach(t IN ITEMS hw_gpu_blend_sweep hw_gpu_pixel_directed hw_gpu_pixel_random)
    add_test(NAME pixel_gpu_${t} COMMAND coregs_pixel_gpu_tests ${t})
    set_tests_properties(pixel_gpu_${t} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

if(WIN32)
    add_executable(coregs_writeback_tests "${COREGS_SRC}/writeback/tests/hw_writeback_tests.cpp")
    target_link_libraries(coregs_writeback_tests PRIVATE coregs_gpu coregs_core coregs_vulkan coregs_profile_gt4)
    foreach(t IN ITEMS
            hw_flat_provoking hw_sprite_corners hw_present_scheduler hw_present_scheduler_hold
            hw_present_scheduler_rate hw_aux_frame_guest_untouched hw_pass_timings hw_gpu_coverage_vectors
            hw_gpu_equivalence_flat hw_gpu_equivalence_texture hw_gpu_equivalence_region
            hw_gpu_equivalence_bilinear hw_gpu_equivalence_tfx_fog hw_gpu_equivalence_depth_alpha
            hw_gpu_equivalence_scissor hw_gpu_raster_blend hw_gpu_raster_pabe hw_gpu_raster_atest
            hw_gpu_raster_date hw_gpu_raster_frame hw_gpu_gouraud_report
            hw_gpu_writeback_aliased_depth hw_readback_pending_draw hw_replay_parity_double_buffer
            hw_presentation_field_phase hw_cache_equals_writeback hw_scale_exact hw_postfx_fxaa
            hw_scale_display_shadow hw_scale_fullscreen_passes hw_scale_interlaced_present
            hw_core2_interlaced_present hw_core2_cpu_sprites hw_core2_gpu_clut hw_core2_palette_segments
            hw_core2_tone_early_drop hw_core2_drawn_page_test hw_core2_width_grace hw_core2_glm_memo
            hw_core2_patch_bits hw_core2_patch_blocks hw_core2_over_target_memo hw_core2_mipmaps
            hw_core2_present_scale hw_core2_forced_bilinear hw_core2_specialised hw_core2_upload_memo
            hw_core2_depth_clear hw_core2_scanmsk hw_core2_display_memory hw_core2_shader_cache
            hw_core2_prepare_shaders hw_core2_shader_guard hw_core2_whole_clears hw_core2_rop
            hw_core2_road_sheen hw_core2_delayed_downloads hw_core2_compute_fills hw_core2_self_texture
            hw_core2_texture_replacement hw_core2_overlap_order hw_core2_point_quads
            hw_core2_effect_toggles hw_scale_target_extent hw_tone_hle_exact
            hw_glm_decode hw_glm_swizzle hw_glm_clut hw_glm_unswizzle hw_glm_upload_merge
            hw_glm_host_upload)
        add_test(NAME wb_${t} COMMAND coregs_writeback_tests ${t})
        set_tests_properties(wb_${t} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    endforeach()
endif()

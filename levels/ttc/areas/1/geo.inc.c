// 0x0E0003B8
const GeoLayout ttc_geo_0003B8[] = {
   GEO_NODE_SCREEN_AREA(10, SCREEN_WIDTH/2, SCREEN_HEIGHT/2, SCREEN_WIDTH/2, SCREEN_HEIGHT/2),
   GEO_OPEN_NODE(),
      GEO_ZBUFFER(0),
      GEO_OPEN_NODE(),
         GEO_NODE_ORTHO(100),
         GEO_OPEN_NODE(),
            GEO_BACKGROUND_COLOR(0xC7FF),
         GEO_CLOSE_NODE(),
      GEO_CLOSE_NODE(),
      GEO_ZBUFFER(1),
      GEO_OPEN_NODE(),
         GEO_CAMERA_FRUSTUM_WITH_FUNC(45, 100, 12800, geo_camera_fov),
         GEO_OPEN_NODE(),
            GEO_CAMERA(2, 0, 2000, 6000, 0, 0, 0, geo_camera_main),
            GEO_OPEN_NODE(),
               GEO_ASM(   0, geo_movtex_pause_control),
               GEO_ASM(0x1400, geo_movtex_update_horizontal),
               GEO_ASM(0x1401, geo_movtex_update_horizontal),
               GEO_DISPLAY_LIST(LAYER_OPAQUE, ttc_seg7_dl_0700AD38),
               GEO_DISPLAY_LIST(LAYER_TRANSPARENT, ttc_seg7_dl_0700B1D8),
               GEO_DISPLAY_LIST(LAYER_ALPHA, ttc_seg7_dl_0700E878),
               GEO_RENDER_OBJ(),
               GEO_ASM(   0, geo_envfx_main),
            GEO_CLOSE_NODE(),
         GEO_CLOSE_NODE(),
      GEO_CLOSE_NODE(),
   GEO_CLOSE_NODE(),
   GEO_END(),
};

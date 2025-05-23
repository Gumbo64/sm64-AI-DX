#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sm64.h"

#include "pc/lua/smlua.h"
#include "pc/lua/utils/smlua_text_utils.h"
#include "game/memory.h"
#include "audio/external.h"

#include "network/network.h"
#include "lua/smlua.h"

#include "audio/audio_api.h"
#include "audio/audio_sdl.h"
#include "audio/audio_null.h"

#include "rom_assets.h"
#include "rom_checker.h"
#include "pc_main.h"
#include "loading.h"
#include "cliopts.h"
#include "configfile.h"
#include "controller/controller_api.h"
#include "controller/controller_keyboard.h"
#include "fs/fs.h"

#include "game/display.h" // for gGlobalTimer
#include "game/game_init.h"
#include "game/main.h"
#include "game/rumble_init.h"

#include "pc/lua/utils/smlua_audio_utils.h"

#include "pc/network/version.h"
#include "pc/network/socket/socket.h"
#include "pc/network/network_player.h"
#include "pc/update_checker.h"
#include "pc/djui/djui.h"
#include "pc/djui/djui_unicode.h"
#include "pc/djui/djui_panel.h"
#include "pc/djui/djui_panel_modlist.h"
#include "pc/djui/djui_fps_display.h"
#include "pc/debuglog.h"
#include "pc/utils/misc.h"

#include "pc/mods/mods.h"

#include "debug_context.h"
#include "menu/intro_geo.h"

#include "gfx_dimensions.h"
#include "game/segment2.h"

#ifdef DISCORD_SDK
#include "pc/discord/discord.h"
#endif

#include "pc/mumble/mumble.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

extern Vp D_8032CF00;

OSMesg D_80339BEC;
OSMesgQueue gSIEventMesgQueue;

s8 gResetTimer;
s8 D_8032C648;
s8 gDebugLevelSelect;
s8 gShowProfiler;
s8 gShowDebugText;

s32 gRumblePakPfs;
u32 gNumVblanks = 0;

u8 gRenderingInterpolated = 0;
f32 gRenderingDelta = 0;

#define FRAMERATE 30
static const f64 sFrameTime = (1.0 / ((double)FRAMERATE));
static f64 sFrameTargetTime = 0;
static f64 sFrameTimeStart;

bool gGameInited = false;
bool gGfxInited = false;

u8 gLuaVolumeMaster = 127;
u8 gLuaVolumeLevel = 127;
u8 gLuaVolumeSfx = 127;
u8 gLuaVolumeEnv = 127;

static struct AudioAPI *audio_api;
struct GfxWindowManagerAPI *wm_api = &WAPI;

extern void gfx_run(Gfx *commands);
extern void thread5_game_loop(void *arg);
extern void create_next_audio_buffer(s16 *samples, u32 num_samples);
void game_loop_one_iteration(void);

void dispatch_audio_sptask(UNUSED struct SPTask *spTask) {}
void set_vblank_handler(UNUSED s32 index, UNUSED struct VblankHandler *handler, UNUSED OSMesgQueue *queue, UNUSED OSMesg *msg) {}

void send_display_list(struct SPTask *spTask) {
    if (!gGameInited) { return; }
    gfx_run((Gfx *)spTask->task.t.data_ptr);
}

#ifdef VERSION_EU
#define SAMPLES_HIGH 560 // gAudioBufferParameters.maxAiBufferLength
#define SAMPLES_LOW 528 // gAudioBufferParameters.minAiBufferLength
#else
#define SAMPLES_HIGH 544
#define SAMPLES_LOW 528
#endif

extern void patch_mtx_before(void);
extern void patch_screen_transition_before(void);
extern void patch_title_screen_before(void);
extern void patch_dialog_before(void);
extern void patch_hud_before(void);
extern void patch_paintings_before(void);
extern void patch_bubble_particles_before(void);
extern void patch_snow_particles_before(void);
extern void patch_djui_before(void);
extern void patch_djui_hud_before(void);
extern void patch_scroll_targets_before(void);

extern void patch_mtx_interpolated(f32 delta);
extern void patch_screen_transition_interpolated(f32 delta);
extern void patch_title_screen_interpolated(f32 delta);
extern void patch_dialog_interpolated(f32 delta);
extern void patch_hud_interpolated(f32 delta);
extern void patch_paintings_interpolated(f32 delta);
extern void patch_bubble_particles_interpolated(f32 delta);
extern void patch_snow_particles_interpolated(f32 delta);
extern void patch_djui_interpolated(f32 delta);
extern void patch_djui_hud(f32 delta);
extern void patch_scroll_targets_interpolated(f32 delta);

static void patch_interpolations_before(void) {
    patch_mtx_before();
    patch_screen_transition_before();
    patch_title_screen_before();
    patch_dialog_before();
    patch_hud_before();
    patch_paintings_before();
    patch_bubble_particles_before();
    patch_snow_particles_before();
    patch_djui_before();
    patch_djui_hud_before();
    patch_scroll_targets_before();
}

static inline void patch_interpolations(f32 delta) {
    patch_mtx_interpolated(delta);
    patch_screen_transition_interpolated(delta);
    patch_title_screen_interpolated(delta);
    patch_dialog_interpolated(delta);
    patch_hud_interpolated(delta);
    patch_paintings_interpolated(delta);
    patch_bubble_particles_interpolated(delta);
    patch_snow_particles_interpolated(delta);
    patch_djui_interpolated(delta);
    patch_djui_hud(delta);
    patch_scroll_targets_interpolated(delta);
}

void produce_interpolation_frames_and_delay(void) {
    u64 frames = 0;
    f64 curTime = clock_elapsed_f64();

    gRenderingInterpolated = true;

    // sanity check target time to deal with hangs and such
    if (fabs(sFrameTargetTime - curTime) > 1) { sFrameTargetTime = curTime - 0.01f; }

    // interpolate and render
    // while ((curTime = clock_elapsed_f64()) < sFrameTargetTime) {
    gfx_start_frame();
    f32 delta = ((!configUncappedFramerate && configFrameLimit == FRAMERATE)
        ? 1.0f
        : MAX(MIN((curTime - sFrameTimeStart) / (sFrameTargetTime - sFrameTimeStart), 1.0f), 0.0f)
    );
    gRenderingDelta = delta;
    if (!gSkipInterpolationTitleScreen) { patch_interpolations(delta); }
    send_display_list(gGfxSPTask);
    gfx_end_frame();

    // delay
    if (!configUncappedFramerate) {
        f64 targetDelta = 1.0 / (f64) configFrameLimit;
        f64 now = clock_elapsed_f64();
        f64 actualDelta = now - curTime;
        if (actualDelta < targetDelta) {
            f64 delay = ((targetDelta - actualDelta) * 1000.0);
            if (delay > 0.0f) { WAPI.delay((u32) delay); }
        }
    }

    frames++;
    // }

    static u64 sFramesSinceFpsUpdate = 0;
    static u64 sLastFpsUpdateTime = 0;

    sFramesSinceFpsUpdate += frames;

    u64 sCurrentFpsUpdateTime = (u64)clock_elapsed_f64();
    if (sLastFpsUpdateTime != sCurrentFpsUpdateTime) {
        u32 fps = sFramesSinceFpsUpdate / (sCurrentFpsUpdateTime - sLastFpsUpdateTime);
        sLastFpsUpdateTime = sCurrentFpsUpdateTime;
        sFramesSinceFpsUpdate = 0;

        djui_fps_display_update(fps);
    }

    sFrameTimeStart = sFrameTargetTime;
    sFrameTargetTime += sFrameTime;
    gRenderingInterpolated = false;
}

inline static void buffer_audio(void) {
    bool shouldMute = configMuteFocusLoss && !WAPI.has_focus();
    const f32 masterMod = (f32)configMasterVolume / 127.0f * (f32)gLuaVolumeMaster / 127.0f;
    set_sequence_player_volume(SEQ_PLAYER_LEVEL, shouldMute ? 0 : (f32)configMusicVolume / 127.0f * (f32)gLuaVolumeLevel / 127.0f * masterMod);
    set_sequence_player_volume(SEQ_PLAYER_SFX,   shouldMute ? 0 : (f32)configSfxVolume / 127.0f * (f32)gLuaVolumeSfx / 127.0f * masterMod);
    set_sequence_player_volume(SEQ_PLAYER_ENV,   shouldMute ? 0 : (f32)configEnvVolume / 127.0f * (f32)gLuaVolumeEnv / 127.0f * masterMod);

    int samplesLeft = audio_api->buffered();
    u32 numAudioSamples = samplesLeft < audio_api->get_desired_buffered() ? SAMPLES_HIGH : SAMPLES_LOW;
    s16 audioBuffer[SAMPLES_HIGH * 2 * 2];
    for (s32 i = 0; i < 2; i++) {
        create_next_audio_buffer(audioBuffer + i * (numAudioSamples * 2), numAudioSamples);
    }
    audio_api->play((u8 *)audioBuffer, 2 * numAudioSamples * 4);
}

void produce_one_frame(void) {
    CTX_EXTENT(CTX_NETWORK, network_update);

    CTX_EXTENT(CTX_INTERP, patch_interpolations_before);

    CTX_EXTENT(CTX_GAME_LOOP, game_loop_one_iteration);

    CTX_EXTENT(CTX_SMLUA, smlua_update);

    CTX_EXTENT(CTX_AUDIO, buffer_audio);

    CTX_EXTENT(CTX_RENDER, produce_interpolation_frames_and_delay);
}

// used for rendering 2D scenes fullscreen like the loading or crash screens
void produce_one_dummy_frame(void (*callback)(), u8 clearColorR, u8 clearColorG, u8 clearColorB) {
    // start frame
    gfx_start_frame();
    config_gfx_pool();
    init_render_image();
    create_dl_ortho_matrix();
    djui_gfx_displaylist_begin();

    // fix scaling issues
    gSPViewport(gDisplayListHead++, VIRTUAL_TO_PHYSICAL(&D_8032CF00));
    gDPSetScissor(gDisplayListHead++, G_SC_NON_INTERLACE, 0, BORDER_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT - BORDER_HEIGHT);

    // clear screen
    create_dl_translation_matrix(MENU_MTX_PUSH, GFX_DIMENSIONS_FROM_LEFT_EDGE(0), 240.f, 0.f);
    create_dl_scale_matrix(MENU_MTX_NOPUSH, (GFX_DIMENSIONS_ASPECT_RATIO * SCREEN_HEIGHT) / 130.f, 3.f, 1.f);
    gDPSetEnvColor(gDisplayListHead++, clearColorR, clearColorG, clearColorB, 0xFF);
    gSPDisplayList(gDisplayListHead++, dl_draw_text_bg_box);
    gSPPopMatrix(gDisplayListHead++, G_MTX_MODELVIEW);

    // call the callback
    callback();

    // render frame
    djui_gfx_displaylist_end();
    end_master_display_list();
    alloc_display_list(0);
    gfx_run((Gfx*) gGfxSPTask->task.t.data_ptr); // send_display_list
    display_and_vsync();
    gfx_end_frame();
}

void audio_shutdown(void) {
    audio_custom_shutdown();
    if (audio_api) {
        if (audio_api->shutdown) audio_api->shutdown();
        audio_api = NULL;
    }
}

void game_deinit(void) {
    if (gGameInited) { configfile_save(configfile_name()); }
    controller_shutdown();
    audio_custom_shutdown();
    audio_shutdown();
    network_shutdown(true, true, false, false);
    smlua_text_utils_shutdown();
    smlua_shutdown();
    smlua_audio_custom_deinit();
    mods_shutdown();
    djui_shutdown();
    gfx_shutdown();
    gGameInited = false;
}

void game_exit(void) {
    LOG_INFO("exiting cleanly");
    game_deinit();
    exit(0);
}

void* main_game_init(UNUSED void* dummy) {
    // load language
    if (!djui_language_init(configLanguage)) { snprintf(configLanguage, MAX_CONFIG_STRING, "%s", ""); }

    LOADING_SCREEN_MUTEX(loading_screen_set_segment_text("Loading"));
    dynos_gfx_init();
    enable_queued_dynos_packs();
    sync_objects_init_system();

    if (gCLIOpts.network != NT_SERVER && !gCLIOpts.skipUpdateCheck) {
        check_for_updates();
    }

    LOADING_SCREEN_MUTEX(loading_screen_set_segment_text("Loading ROM Assets"));
    rom_assets_load();
    smlua_text_utils_init();

    mods_init();
    enable_queued_mods();
    LOADING_SCREEN_MUTEX(
        gCurrLoadingSegment.percentage = 0;
        loading_screen_set_segment_text("Starting Game");
    );

    if (gCLIOpts.fullscreen == 1) { configWindow.fullscreen = true; }
    else if (gCLIOpts.fullscreen == 2) { configWindow.fullscreen = false; }

    audio_init();
    sound_init();
    smlua_audio_custom_init();
    network_player_init();

    gGameInited = true;
}

int main(int argc, char *argv[]) {
    // handle terminal arguments
    if (!parse_cli_opts(argc, argv)) { return 0; }

#ifdef _WIN32
    // handle Windows console
    if (gCLIOpts.console) {
        SetConsoleOutputCP(CP_UTF8);
    } else {
        FreeConsole();
        freopen("NUL", "w", stdout);
    }
#endif

#ifdef _WIN32
    if (gCLIOpts.savePath[0]) {
        char portable_path[SYS_MAX_PATH] = {};
        sys_windows_short_path_from_mbs(portable_path, SYS_MAX_PATH, gCLIOpts.savePath);
        fs_init(portable_path);
    } else {
        fs_init(sys_user_path());
    }
#else
    fs_init(gCLIOpts.savePath[0] ? gCLIOpts.savePath : sys_user_path());
#endif

    configfile_load();

    legacy_folder_handler();

    // create the window almost straight away
    if (!gGfxInited) {
        gfx_init(&WAPI, &RAPI, TITLE);
        WAPI.set_keyboard_callbacks(keyboard_on_key_down, keyboard_on_key_up, keyboard_on_all_keys_up, keyboard_on_text_input);
    }

    // render the rom setup screen
    if (!main_rom_handler()) {
#ifdef LOADING_SCREEN_SUPPORTED
        if (!gCLIOpts.hideLoadingScreen) {
            render_rom_setup_screen(); // holds the game load until a valid rom is provided
        } else
#endif
        {
            printf("ERROR: could not find valid vanilla us sm64 rom in game's user folder\n");
            return 0;
        }
    }

    // start the thread for setting up the game
#ifdef LOADING_SCREEN_SUPPORTED
    bool threadSuccess = false;
    if (!gCLIOpts.hideLoadingScreen && pthread_mutex_init(&gLoadingThreadMutex, NULL) == 0) {
        gIsThreaded = true;
        if (pthread_create(&gLoadingThreadId, NULL, main_game_init, NULL) == 0) {
            render_loading_screen(); // render the loading screen while the game is setup
            threadSuccess = true;
        }
        gIsThreaded = false;
        pthread_mutex_destroy(&gLoadingThreadMutex);
    }
    if (!threadSuccess)
#endif
    {
        main_game_init(NULL); // failsafe incase threading doesn't work
    }

    // initialize sm64 data and controllers
    thread5_game_loop(NULL);

    // initialize sound outside threads
#if defined(AAPI_SDL1) || defined(AAPI_SDL2)
    if (!audio_api && audio_sdl.init()) { audio_api = &audio_sdl; }
#endif
    if (!audio_api) { audio_api = &audio_null; }

#ifdef LOADING_SCREEN_SUPPORTED
    loading_screen_reset();
#endif

    // initialize djui
    djui_init();
    djui_unicode_init();
    djui_init_late();
    djui_console_message_dequeue();

    show_update_popup();

    // initialize network
    if (gCLIOpts.network == NT_CLIENT) {
        network_set_system(NS_SOCKET);
        snprintf(gGetHostName, MAX_CONFIG_STRING, "%s", gCLIOpts.joinIp);
        snprintf(configJoinIp, MAX_CONFIG_STRING, "%s", gCLIOpts.joinIp);
        configJoinPort = gCLIOpts.networkPort;
        network_init(NT_CLIENT, false);
    } else if (gCLIOpts.network == NT_SERVER) {
        configNetworkSystem = NS_SOCKET;
        configHostPort = gCLIOpts.networkPort;

        // horrible, hacky fix for mods that access marioObj straight away
        // best fix: host with the standard main menu method
        static struct Object sHackyObject = { 0 };
        gMarioStates[0].marioObj = &sHackyObject;

        extern void djui_panel_do_host(bool reconnecting, bool playSound);
        djui_panel_do_host(NULL, false);
    } else {
        network_init(NT_NONE, false);
    }

    mumble_init();

    // main loop
//     while (true) {
//         debug_context_reset();
//         CTX_BEGIN(CTX_FRAME);
//         WAPI.main_loop(produce_one_frame);
// #ifdef DISCORD_SDK
//         discord_update();
// #endif
//         mumble_update();
// #ifdef DEBUG
//         fflush(stdout);
//         fflush(stderr);
// #endif
//         CTX_END(CTX_FRAME);
//     }

    return 0;
}

void step_game(int steps,
    int playerIndex, int stickX, int stickY, 
    int buttonA, int buttonB, int buttonX, int buttonY,
    int buttonL, int buttonR, int buttonZ, int buttonStart,
    int buttonDU, int buttonDL, int buttonDR, int buttonDD,
    int buttonCU, int buttonCL, int buttonCR, int buttonCD) {
    for (int i = 0; i < steps; i++) {
        set_controller(playerIndex, stickX, stickY, 
            buttonA, buttonB, buttonX, buttonY,
            buttonL, buttonR, buttonZ, buttonStart,
            buttonDU, buttonDL, buttonDR, buttonDD,
            buttonCU, buttonCL, buttonCR, buttonCD
        );
        debug_context_reset();
        CTX_BEGIN(CTX_FRAME);
        WAPI.main_loop(produce_one_frame);
    #ifdef DISCORD_SDK
        discord_update();
    #endif
        mumble_update();
    #ifdef DEBUG
        fflush(stdout);
        fflush(stderr);
    #endif
        CTX_END(CTX_FRAME);
    }
}

void set_controller(int playerIndex, int stickX, int stickY, 
    int buttonA, int buttonB, int buttonX, int buttonY,
    int buttonL, int buttonR, int buttonZ, int buttonStart,
    int buttonDU, int buttonDL, int buttonDR, int buttonDD,
    int buttonCU, int buttonCL, int buttonCR, int buttonCD
) {
    struct Controller *controller = &gControllers[playerIndex];

    controller->rawStickX = stickX;
    controller->rawStickY = stickY;

    controller->controllerData->button = 0;
    controller->controllerData->button |= buttonA ? A_BUTTON : 0;
    controller->controllerData->button |= buttonB ? B_BUTTON : 0;
    controller->controllerData->button |= buttonX ? X_BUTTON : 0;
    controller->controllerData->button |= buttonY ? Y_BUTTON : 0;
    controller->controllerData->button |= buttonL ? L_TRIG : 0;
    controller->controllerData->button |= buttonR ? R_TRIG : 0;
    controller->controllerData->button |= buttonZ ? Z_TRIG : 0;
    controller->controllerData->button |= buttonStart ? START_BUTTON : 0;
    controller->controllerData->button |= buttonDU ? U_JPAD : 0;
    controller->controllerData->button |= buttonDL ? L_JPAD : 0;
    controller->controllerData->button |= buttonDR ? R_JPAD : 0;
    controller->controllerData->button |= buttonDD ? D_JPAD : 0;
    controller->controllerData->button |= buttonCU ? U_CBUTTONS : 0;
    controller->controllerData->button |= buttonCL ? L_CBUTTONS : 0;
    controller->controllerData->button |= buttonCR ? R_CBUTTONS : 0;
    controller->controllerData->button |= buttonCD ? D_CBUTTONS : 0;

    if ( controller->rawStickX != 0 && controller->rawStickY != 0){
        controller->controllerData->button |= INPUT_NONZERO_ANALOG;
    }

    controller->buttonPressed = controller->controllerData->button
                    & (controller->controllerData->button ^ controller->buttonDown);

    controller->buttonDown = controller->controllerData->button;

    adjust_analog_stick(controller);
}

struct MarioState *get_mario_state(int index) {
    return &gMarioStates[index];
}

struct NetworkPlayer *get_network_player(int index) {
    return &gNetworkPlayers[index];
}



#include "game/camera.h"

Vec3f *get_lakitu_pos() {
    return &gLakituState.pos;
}

s16 get_lakitu_yaw() {
    return gLakituState.yaw;
}

int local_index_to_global(int index) {
    return gNetworkPlayers[index].globalIndex;
}

int global_index_to_local(int index) {
    for (int i = 0; i < MAX_PLAYERS; i++) {
        if (gNetworkPlayers[i].globalIndex == index) {
            return i;
        }
    }
    return -1;
} 



// RAYCASTS //

// f32* = Vec3f
#include "engine/surface_collision.h" // for find_surface_on_ray to work
#include "game/camera.h" // for gCamera
#include "gfx/gfx_pc.h"



void sample_sphere_surface(Vec3f point, f32 new_len) {
    f32 len = 2.0;
    while (len > 1.0 || len == 0.0) {
        point[0] = ((f32)rand()) / RAND_MAX * 2.0 - 1.0;
        point[1] = ((f32)rand()) / RAND_MAX * 2.0 - 1.0;
        point[2] = ((f32)rand()) / RAND_MAX * 2.0 - 1.0;
        len = point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
    }
    len = sqrtf(len);
    f32 scale = new_len / len;
    point[0] *= scale;
    point[1] *= scale;
    point[2] *= scale;
}



bool raycast_with_normal(Vec3f hitpos, Vec3f normal, Vec3f start, Vec3f dir) {
    struct Surface *surf = NULL;
    find_surface_on_ray(start, dir, &surf, hitpos, 3.0f);

    
    if (surf == NULL || hitpos[1] <= -6000) {
        normal[0] = 0;
        normal[1] = 0;
        normal[2] = 0;
        return false;
    } else {
        // printf("surface type: %d\n", surf->type);

        normal[0] = surf->normal.x;
        normal[1] = surf->normal.y;
        normal[2] = surf->normal.z;
        return true;
    }
}

void raycast_sphere_with_normal(Vec3f *hitpos_arr, Vec3f *normal_arr, int amount, 
                                f32 maxRayLength, f32 cameraDirBiasFactor) {

    Vec3f start;
    // vec3f_copy(start, gCamera->pos);
    vec3f_copy(start, gMarioStates[0].pos);
    start[1] += 200.0f;
    
    Vec3f velBias;
    vec3f_copy(velBias, gCamera->focus);
    vec3f_sub(velBias, gCamera->pos);
    vec3f_normalize(velBias);
    velBias[0] *= cameraDirBiasFactor;
    velBias[1] *= cameraDirBiasFactor;
    velBias[2] *= cameraDirBiasFactor;

    for (int i = 0; i < amount; i++) {
        // if (i == amount / 2) {
        //     // sample half of the rays from the player, half from camera
        //     vec3f_copy(start, gMarioStates[0].pos);
        //     start[1] += 200.0f;
        // }

        // take raycasts from a range of starting points
        start[1] += 400.0f / amount;

        Vec3f dir;
        while (true) {
            sample_sphere_surface(dir, 1);

            vec3f_add(dir, velBias);
            dir[0] *= maxRayLength;
            dir[1] *= maxRayLength;
            dir[2] *= maxRayLength;
            
            
            if (raycast_with_normal(hitpos_arr[i], normal_arr[i], start, dir)) {
                break;
            }
        }

    }
}


// void raycast_grid(Vec3f *hitpos_arr, Vec3f *normal_arr, int width, int height, int gridSpacing,
//                   f32 maxRayLength, f32 cameraDirBiasFactor) {
//     Vec3f start;
//     vec3f_copy(start, gMarioStates[0].pos);
//     start[1] += 200.0f;
    
//     Vec3f dir;
//     vec3f_copy(dir, gCamera->focus);
//     vec3f_sub(dir, gCamera->pos);
//     vec3f_normalize(dir);

    
//     for (int i = -height; i < height; i++) {
//         for (int j = -width; j < width; j++) {
//             Vec3f pos;
//             pos[0] = sin(j * (M_PI / width)) * gridSpacing;
//             pos[1] = sin(i * (M_PI / height)) * gridSpacing;
//             pos[2] = 0;

//             vec3f_add(pos, gMarioStates[0].pos);
            
            

//             if (raycast_with_normal(hitpos_arr[i * width + j], normal_arr[i * width + j], start, dir)) {
//                 break;
//             }
            
//         }
//     }
// }
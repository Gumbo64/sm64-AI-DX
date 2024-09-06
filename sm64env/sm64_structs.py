import ctypes


# struct MarioState
# {
#     /*0x00*/ u16 playerIndex;
#     /*0x02*/ u16 input;
#     /*0x04*/ u32 flags;
#     /*0x08*/ u32 particleFlags;
#     /*0x0C*/ u32 action;
#     /*0x10*/ u32 prevAction;
#     /*0x14*/ u32 terrainSoundAddend;
#     /*0x18*/ u16 actionState;
#     /*0x1A*/ u16 actionTimer;
#     /*0x1C*/ u32 actionArg;
#     /*0x20*/ f32 intendedMag;
#     /*0x24*/ s16 intendedYaw;
#     /*0x26*/ s16 invincTimer;
#     /*0x28*/ u8 framesSinceA;
#     /*0x29*/ u8 framesSinceB;
#     /*0x2A*/ u8 wallKickTimer;
#     /*0x2B*/ u8 doubleJumpTimer;
#     /*0x2C*/ Vec3s faceAngle;
#     /*0x32*/ Vec3s angleVel;
#     /*0x38*/ s16 slideYaw;
#     /*0x3A*/ s16 twirlYaw;
#     /*0x3C*/ Vec3f pos;
#     /*0x48*/ Vec3f vel;
#     /*0x54*/ f32 forwardVel;
#     /*0x58*/ f32 slideVelX;
#     /*0x5C*/ f32 slideVelZ;
#     /*0x60*/ struct Surface *wall;
#     /*0x64*/ struct Surface *ceil;
#     /*0x68*/ struct Surface *floor;
#     /*0x6C*/ f32 ceilHeight;
#     /*0x70*/ f32 floorHeight;
#     /*0x74*/ s16 floorAngle;
#     /*0x76*/ s16 waterLevel;
#     /*0x78*/ struct Object *interactObj;
#     /*0x7C*/ struct Object *heldObj;
#     /*0x80*/ struct Object *usedObj;
#     /*0x84*/ struct Object *riddenObj;
#     /*0x88*/ struct Object *marioObj;
#     /*0x8C*/ struct SpawnInfo *spawnInfo;
#     /*0x90*/ struct Area *area;
#     /*0x94*/ struct PlayerCameraState *statusForCamera;
#     /*0x98*/ struct MarioBodyState *marioBodyState;
#     /*0x9C*/ struct Controller *controller;
#     /*0xA0*/ struct MarioAnimation *animation;
#     /*0xA4*/ u32 collidedObjInteractTypes;
#     /*0xA8*/ s16 numCoins;
#     /*0xAA*/ s16 numStars;
#     /*0xAC*/ s8 numKeys; // Unused key mechanic
#     /*0xAD*/ s8 numLives;
#     /*0xAE*/ s16 health;
#     /*0xB0*/ s16 unkB0;
#     /*0xB2*/ u8 hurtCounter;
#     /*0xB3*/ u8 healCounter;
#     /*0xB4*/ u8 squishTimer;
#     /*0xB5*/ u8 fadeWarpOpacity;
#     /*0xB6*/ u16 capTimer;
#     /*0xB8*/ s16 prevNumStarsForDialog;
#     /*0xBC*/ f32 peakHeight;
#     /*0xC0*/ f32 quicksandDepth;
#     /*0xC4*/ f32 unkC4;
#     /*0xC8*/ s16 currentRoom;
#     /*0xCA*/ struct Object* heldByObj;
#     /*????*/ u8 isSnoring;
#     /*????*/ struct Object* bubbleObj;
#     /*????*/ u8 freeze;

#     // Variables for a spline curve animation (used for the flight path in the grand star cutscene)
#     /*????*/ Vec4s* splineKeyframe;
#     /*????*/ f32 splineKeyframeFraction;
#     /*????*/ s32 splineState;

#     /*????*/ Vec3f nonInstantWarpPos;
#     /*????*/ struct Character* character;
#     /*????*/ u8 wasNetworkVisible;
#     /*????*/ f32 minimumBoneY;
#     /*????*/ f32 curAnimOffset;
#     /*????*/ u8 knockbackTimer;
#     /*????*/ u8 specialTripleJump;
#     /*????*/ Vec3f wallNormal;
#     /*????*/ u8 visibleToEnemies;
#     /*????*/ u32 cap;
#     /*????*/ u8 bounceSquishTimer;
#     /*????*/ u8 skipWarpInteractionsTimer;
#     /*????*/ s16 dialogId;
# };

class MarioState(ctypes.Structure):
    _fields_ = [
        ("playerIndex", ctypes.c_ushort),
        ("input", ctypes.c_ushort),

        ("flags", ctypes.c_uint),
        ("particleFlags", ctypes.c_uint),
        ("action", ctypes.c_uint),
        ("prevAction", ctypes.c_uint),
        ("terrainSoundAddend", ctypes.c_uint),

        ("actionState", ctypes.c_ushort),
        ("actionTimer", ctypes.c_ushort),
        
        ("actionArg", ctypes.c_uint),
        ("intendedMag", ctypes.c_float),
        ("intendedYaw", ctypes.c_short),
        ("invincTimer", ctypes.c_short),
        ("framesSinceA", ctypes.c_ubyte),
        ("framesSinceB", ctypes.c_ubyte),
        ("wallKickTimer", ctypes.c_ubyte),
        ("doubleJumpTimer", ctypes.c_ubyte),
        ("faceAngle", ctypes.c_short * 3),
        ("angleVel", ctypes.c_short * 3),
        ("slideYaw", ctypes.c_short),
        ("twirlYaw", ctypes.c_short),
        ("pos", ctypes.c_float * 3),
        ("vel", ctypes.c_float * 3),
        ("forwardVel", ctypes.c_float),
        ("slideVelX", ctypes.c_float),
        ("slideVelZ", ctypes.c_float),
        ("wall", ctypes.c_void_p),
        ("ceil", ctypes.c_void_p),
        ("floor", ctypes.c_void_p),
        ("ceilHeight", ctypes.c_float),
        ("floorHeight", ctypes.c_float),
        ("floorAngle", ctypes.c_short),
        ("waterLevel", ctypes.c_short),
        ("interactObj", ctypes.c_void_p),
        ("heldObj", ctypes.c_void_p),
        ("usedObj", ctypes.c_void_p),
        ("riddenObj", ctypes.c_void_p),
        ("marioObj", ctypes.c_void_p),
        ("spawnInfo", ctypes.c_void_p),
        ("area", ctypes.c_void_p),
        ("statusForCamera", ctypes.c_void_p),
        ("marioBodyState", ctypes.c_void_p),
        ("controller", ctypes.c_void_p),
        ("animation", ctypes.c_void_p),
        ("collidedObjInteractTypes", ctypes.c_uint),
        ("numCoins", ctypes.c_short),
        ("numStars", ctypes.c_short),
        ("numKeys", ctypes.c_byte),
        ("numLives", ctypes.c_byte),
        ("health", ctypes.c_short),
        ("unkB0", ctypes.c_short),
        ("hurtCounter", ctypes.c_byte),
        ("healCounter", ctypes.c_byte),
        ("squishTimer", ctypes.c_byte),
        ("fadeWarpOpacity", ctypes.c_byte),
        ("capTimer", ctypes.c_ushort),
        ("prevNumStarsForDialog", ctypes.c_short),
        ("peakHeight", ctypes.c_float),
        ("quicksandDepth", ctypes.c_float),
        ("unkC4", ctypes.c_float),
        ("currentRoom", ctypes.c_short),
        ("heldByObj", ctypes.c_void_p),
        ("isSnoring", ctypes.c_byte),
        ("bubbleObj", ctypes.c_void_p),
        ("freeze", ctypes.c_byte),
        ("splineKeyframe", ctypes.c_void_p),
        ("splineKeyframeFraction", ctypes.c_float),
        ("splineState", ctypes.c_int),
        ("nonInstantWarpPos", ctypes.c_float * 3),
        ("character", ctypes.c_void_p),
        ("wasNetworkVisible", ctypes.c_byte),
        ("minimumBoneY", ctypes.c_float),
        ("curAnimOffset", ctypes.c_float),
        ("knockbackTimer", ctypes.c_byte),
        ("specialTripleJump", ctypes.c_byte),
        ("wallNormal", ctypes.c_float * 3),
        ("visibleToEnemies", ctypes.c_byte),
        ("cap", ctypes.c_uint),
        ("bounceSquishTimer", ctypes.c_byte),
        ("skipWarpInteractionsTimer", ctypes.c_byte),
        ("dialogId", ctypes.c_short),
    ]


# struct NetworkPlayer {
#     bool connected;
#     u8 type;
#     u8 localIndex;
#     u8 globalIndex;
#     bool moderator;
#     f32 lastReceived;
#     f32 lastSent;
#     f32 lastPingSent;
#     u16 currLevelAreaSeqId;
#     s16 currCourseNum;
#     s16 currActNum;
#     s16 currLevelNum;
#     s16 currAreaIndex;
#     bool currLevelSyncValid;
#     bool currAreaSyncValid;
#     bool currPositionValid;
#     u8 fadeOpacity;
#     u8 onRxSeqId;
#     u8 modelIndex;
#     u8 gag;
#     u32 ping;
#     struct PlayerPalette palette;
#     char name[MAX_CONFIG_STRING];

#     char description[MAX_DESCRIPTION_STRING];
#     u8 descriptionR;
#     u8 descriptionG;
#     u8 descriptionB;
#     u8 descriptionA;

#     u8 overrideModelIndex;
#     struct PlayerPalette overridePalette;

#     u16 rxSeqIds[MAX_RX_SEQ_IDS];
#     u32 rxPacketHash[MAX_RX_SEQ_IDS];

#     char discordId[64];

#     // legacy fields to allow mods not to fully break (they don't do anything anymore)
#     u8 paletteIndex;
#     u8 overridePaletteIndex;
#     u8 overridePaletteIndexLp;
# };

class NetworkPlayer(ctypes.Structure):
    _fields_ = [
        ("connected", ctypes.c_bool),

        ("type", ctypes.c_ubyte),
        ("localIndex", ctypes.c_ubyte),
        ("globalIndex", ctypes.c_ubyte),

        ("moderator", ctypes.c_bool),

        ("lastReceived", ctypes.c_float),
        ("lastSent", ctypes.c_float),
        ("lastPingSent", ctypes.c_float),

        ("currLevelAreaSeqId", ctypes.c_ushort),
        ("currCourseNum", ctypes.c_short),
        ("currActNum", ctypes.c_short),
        ("currLevelNum", ctypes.c_short),
        ("currAreaIndex", ctypes.c_short),

        ("currLevelSyncValid", ctypes.c_bool),
        ("currAreaSyncValid", ctypes.c_bool),
        ("currPositionValid", ctypes.c_bool),

        ("fadeOpacity", ctypes.c_ubyte),
        ("onRxSeqId", ctypes.c_ubyte),
        ("modelIndex", ctypes.c_ubyte),
        ("gag", ctypes.c_ubyte),

        ("ping", ctypes.c_uint),

        ("palette", ctypes.c_ubyte * 3 * 8), # struct PlayerPalette

        ("name", ctypes.c_char * 64),
        ("description", ctypes.c_char * 20),

        ("descriptionR", ctypes.c_ubyte),
        ("descriptionG", ctypes.c_ubyte),
        ("descriptionB", ctypes.c_ubyte),
        ("descriptionA", ctypes.c_ubyte),
        ("overrideModelIndex", ctypes.c_ubyte),

        ("overridePalette", ctypes.c_ubyte * 3 * 8), # struct PlayerPalette
        ("rxSeqIds", ctypes.c_ushort * 256),
        ("rxPacketHash", ctypes.c_uint * 256),

        ("discordId", ctypes.c_char * 64),

        ("paletteIndex", ctypes.c_ubyte),
        ("overridePaletteIndex", ctypes.c_ubyte),
        ("overridePaletteIndexLp", ctypes.c_ubyte),
    ]
    

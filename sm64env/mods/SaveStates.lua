-- name: SaveStates
-- description: \\#ff4040\\SaveStates \\#ff8040\\v2.2.1 \\#ffff40\\\by \\#40ff40\\iZePlayz\n\n\\#80ffff\\This mod adds SaveStates to SM64COOPDX and SM64EX-COOP!\n\n\\#ff80ff\\-Save up to 4 states at the same time by using the 4 D-Pad buttons\n-To load states, hold down L and press the corresponding D-Pad button\n-Each player gets their own 4 local savestate slots (Global SaveStates maybe in the future)\n\n\\#8080ff\\-Use /autoload or /al to toggle AutoLoading the latest savestate upon death\n-Use /autoheal or /ah to toggle AutoHealing after loading savestates\n-Use /savestates, /savestate, or /ss to get infos about your local settings and slots

local autoload_enabled = false
local autoheal_enabled = false
local latest_savestate = nil
local wait_until_inited = nil
local ready_to_load = nil

function default()
    return { nil, nil, nil, nil }
end

local savestates = {
    gNetworkPlayers = {
        currLevelNum = default(),
        currActNum = default(),
        currAreaIndex = default()
    },
    action = default(),
    actionArg = default(),
    actionState = default(),
    actionTimer = default(),
    angleVel = {
        x = default(),
        y = default(),
        z = default()
    },
    animation = {
        targetAnim = default()
    },
    area = {
        camera = default(),
        flags = default(),
        index = default(),
        instantWarps = default(),
        musicParam = default(),
        musicParam2 = default(),
        numRedCoins = default(),
        numSecrets = default(),
        objectSpawnInfos = {
            activeAreaIndex = default(),
            areaIndex = default(),
            behaviorArg = default(),
            startAngle = {
                x = default(),
                y = default(),
                z = default()
            },
            startPos = {
                x = default(),
                y = default(),
                z = default()
            },
            unk18 = {
                extraFlags = default(),
                flags = default()
            }
        },
        paintingWarpNodes = {
            destArea = default(),
            destLevel = default(),
            destNode = default(),
            id = default()
        },
        terrainType = default(),
        warpNodes = {
            next = default(),
            node = {
                destArea = default(),
                destLevel = default(),
                destNode = default(),
                id = default()
            },
            object = default()
        }
    },
    bounceSquishTimer = default(),
    bubbleObj = default(),
    cap = default(),
    capTimer = default(),
    ceil = default(),
    ceilHeight = default(),
    character = default(),
    collidedObjInteractTypes = default(),
    controller = {
        buttonDown = default(),
        buttonPressed = default(),
        extStickX = default(),
        extStickY = default(),
        port = default(),
        rawStickX = default(),
        rawStickY = default(),
        stickMag = default(),
        stickX = default(),
        stickY = default()
    },
    curAnimOffset = default(),
    currentRoom = default(),
    doubleJumpTimer = default(),
    faceAngle = {
        x = default(),
        y = default(),
        z = default()
    },
    fadeWarpOpacity = default(),
    flags = default(),
    floor = default(),
    floorAngle = default(),
    floorHeight = default(),
    forwardVel = default(),
    framesSinceA = default(),
    framesSinceB = default(),
    freeze = default(),
    healCounter = default(),
    health = default(),
    heldByObj = default(),
    heldObj = default(),
    hurtCounter = default(),
    input = default(),
    intendedMag = default(),
    intendedYaw = default(),
    interactObj = default(),
    invincTimer = default(),
    isSnoring = default(),
    knockbackTimer = default(),
    marioBodyState = {
        action = default(),
        capState = default(),
        eyeState = default(),
        grabPos = default(),
        handState = default(),
        headAngle = {
            x = default(),
            y = default(),
            z = default()
        },
        headPos = {
            x = default(),
            y = default(),
            z = default()
        },
        heldObjLastPosition = {
            x = default(),
            y = default(),
            z = default()
        },
        modelState = default(),
        punchState = default(),
        torsoAngle = {
            x = default(),
            y = default(),
            z = default()
        },
        torsoPos = {
            x = default(),
            y = default(),
            z = default()
        },
        wingFlutter = default()
    },
    marioObj = {
        activeFlags = default(),
        areaTimer = default(),
        areaTimerDuration = default(),
        areaTimerType = default(),
        bhvDelayTimer = default(),
        collidedObjInteractTypes = default(),
        collisionData = default(),
        ctx = default(),
        globalPlayerIndex = default(),
        header = {
            gfx = {
                activeAreaIndex = default(),
                areaIndex = default(),
                disableAutomaticShadowPos = default(),
                shadowInvisible = default(),
                skipInViewCheck = default()
            }
        },
        heldByPlayerIndex = default(),
        hitboxDownOffset = default(),
        hitboxHeight = default(),
        hitboxRadius = default(),
        hookRender = default(),
        hurtboxHeight = default(),
        hurtboxRadius = default(),
        parentObj = default(),
        platform = default(),
        prevObj = default(),
        setHome = default(),
        unused1 = default(),
        usingObj = default()
    },
    minimumBoneY = default(),
    nonInstantWarpPos = {
        x = default(),
        y = default(),
        z = default()
    },
    numCoins = default(),
    numKeys = default(),
    numLives = default(),
    numStars = default(),
    particleFlags = default(),
    peakHeight = default(),
    pos = {
        x = default(),
        y = default(),
        z = default()
    },
    prevAction = default(),
    prevNumStarsForDialog = default(),
    quicksandDepth = default(),
    riddenObj = default(),
    slideVelX = default(),
    slideVelZ = default(),
    slideYaw = default(),
    spawnInfo = default(),
    specialTripleJump = default(),
    splineKeyframe = default(),
    splineKeyframeFraction = default(),
    splineState = default(),
    squishTimer = default(),
    statusForCamera = {
        action = default(),
        cameraEvent = default(),
        faceAngle = {
            x = default(),
            y = default(),
            z = default()
        },
        headRotation = {
            x = default(),
            y = default(),
            z = default()
        },
        pos = {
            x = default(),
            y = default(),
            z = default()
        },
        unused = default(),
        usedObj = default()
    },
    terrainSoundAddend = default(),
    twirlYaw = default(),
    unkB0 = default(),
    unkC4 = default(),
    usedObj = default(),
    vel = {
        x = default(),
        y = default(),
        z = default()
    },
    wall = default(),
    wallKickTimer = default(),
    wallNormal = {
        x = default(),
        y = default(),
        z = default()
    },
    wasNetworkVisible = default(),
    waterLevel = default()
}

function on_join(m)
    if m.playerIndex == 0 then
        reset_all(m)
    end
end

function on_leave(m)
    if m.playerIndex == 0 then
        reset_all(m)
    end
end

function on_death(m)
    if m.playerIndex == 0 then
        if autoload_enabled == true then
            if latest_savestate ~= nil and is_state_slot_not_empty(latest_savestate) then
                local nCoins = m.numCoins
                local nRedCoins = m.area.numRedCoins
                local nSecrets = m.area.numSecrets
                init_single_mario(m)
                m.pos.x = 0
                m.pos.y = 0
                m.pos.z = 0
                m.health = 0x880
                m.numLives = m.numLives + 1
                m.numCoins = nCoins
                m.area.numRedCoins = nRedCoins
                m.area.numSecrets = nSecrets
                soft_reset_camera(m.area.camera)
                djui_popup_create("\\#ff80ff\\[AutoLoad] Loading latest state...", 1)
                load_state_slot(m, latest_savestate)
                return false
            else
                latest_savestate = nil
                play_sound(SOUND_GENERAL_FLAME_OUT, m.marioObj.header.gfx.cameraToObject)
                djui_popup_create("\\#a0a0a0\\[AutoLoad] 404 latest state not found!", 1)
                return true
            end
        else
            return true
        end
    end
end

function on_level_init()
    if wait_until_inited ~= nil then
        ready_to_load = wait_until_inited
        wait_until_inited = nil
    end
end

function on_mario_update(m)
    if m.playerIndex == 0 then
        if ready_to_load ~= nil then
            load_state_slot(m, ready_to_load)
        else
            if (m.controller.buttonDown & L_TRIG) == 0 then
                if (m.controller.buttonPressed & L_JPAD) ~= 0 then
                    save_state_slot(m, 0, "\\#ff4040\\")
                elseif (m.controller.buttonPressed & U_JPAD) ~= 0 then
                    save_state_slot(m, 1, "\\#40ff40\\")
                elseif (m.controller.buttonPressed & R_JPAD) ~= 0 then
                    save_state_slot(m, 2, "\\#6060ff\\")
                elseif (m.controller.buttonPressed & D_JPAD) ~= 0 then
                    save_state_slot(m, 3, "\\#ffff40\\")
                end
            else
                if (m.controller.buttonPressed & L_JPAD) ~= 0 then
                    load_state_slot(m, 0)
                elseif (m.controller.buttonPressed & U_JPAD) ~= 0 then
                    load_state_slot(m, 1)
                elseif (m.controller.buttonPressed & R_JPAD) ~= 0 then
                    load_state_slot(m, 2)
                elseif (m.controller.buttonPressed & D_JPAD) ~= 0 then
                    load_state_slot(m, 3)
                end
            end
        end
    end
end

function reset_all(m)
    delete_state_slot(m, 0)
    delete_state_slot(m, 1)
    delete_state_slot(m, 2)
    delete_state_slot(m, 3)
    autoload_enabled = false
    autoheal_enabled = false
    latest_savestate = nil
end

function save_state_slot(m, slot, color)
    savestates.gNetworkPlayers.currLevelNum[slot] = gNetworkPlayers[0].currLevelNum
    savestates.gNetworkPlayers.currActNum[slot] = gNetworkPlayers[0].currActNum
    savestates.gNetworkPlayers.currAreaIndex[slot] = gNetworkPlayers[0].currAreaIndex

    if m then
        savestates.action[slot] = m.action or nil
        savestates.actionArg[slot] = m.actionArg or nil
        savestates.actionState[slot] = m.actionState or nil
        savestates.actionTimer[slot] = m.actionTimer or nil

        if m.angleVel then
            savestates.angleVel.x[slot] = m.angleVel.x or nil
            savestates.angleVel.y[slot] = m.angleVel.y or nil
            savestates.angleVel.z[slot] = m.angleVel.z or nil
        end

        if m.animation and m.animation.targetAnim then
            savestates.animation.targetAnim[slot] = m.animation.targetAnim or nil
        end

        if m.area then
            savestates.area.camera[slot] = m.area.camera or nil
            savestates.area.flags[slot] = m.area.flags or nil
            savestates.area.index[slot] = m.area.index or nil
            savestates.area.instantWarps[slot] = m.area.instantWarps or nil
            savestates.area.musicParam[slot] = m.area.musicParam or nil
            savestates.area.musicParam2[slot] = m.area.musicParam2 or nil
            savestates.area.numRedCoins[slot] = m.area.numRedCoins or nil
            savestates.area.numSecrets[slot] = m.area.numSecrets or nil

            if m.area.objectSpawnInfos then
                savestates.area.objectSpawnInfos.activeAreaIndex[slot] = m.area.objectSpawnInfos.activeAreaIndex or nil
                savestates.area.objectSpawnInfos.areaIndex[slot] = m.area.objectSpawnInfos.areaIndex or nil
                savestates.area.objectSpawnInfos.behaviorArg[slot] = m.area.objectSpawnInfos.behaviorArg or nil

                if m.area.objectSpawnInfos.startAngle then
                    savestates.area.objectSpawnInfos.startAngle.x[slot] = m.area.objectSpawnInfos.startAngle.x or nil
                    savestates.area.objectSpawnInfos.startAngle.y[slot] = m.area.objectSpawnInfos.startAngle.y or nil
                    savestates.area.objectSpawnInfos.startAngle.z[slot] = m.area.objectSpawnInfos.startAngle.z or nil
                end

                if m.area.objectSpawnInfos.startPos then
                    savestates.area.objectSpawnInfos.startPos.x[slot] = m.area.objectSpawnInfos.startPos.x or nil
                    savestates.area.objectSpawnInfos.startPos.y[slot] = m.area.objectSpawnInfos.startPos.y or nil
                    savestates.area.objectSpawnInfos.startPos.z[slot] = m.area.objectSpawnInfos.startPos.z or nil
                end

                if m.area.objectSpawnInfos.unk18 then
                    savestates.area.objectSpawnInfos.unk18.extraFlags[slot] = m.area.objectSpawnInfos.unk18.extraFlags or nil
                    savestates.area.objectSpawnInfos.unk18.flags[slot] = m.area.objectSpawnInfos.unk18.flags or nil
                end
            end

            if m.area.paintingWarpNodes then
                savestates.area.paintingWarpNodes.destArea[slot] = m.area.paintingWarpNodes.destArea or nil
                savestates.area.paintingWarpNodes.destLevel[slot] = m.area.paintingWarpNodes.destLevel or nil
                savestates.area.paintingWarpNodes.destNode[slot] = m.area.paintingWarpNodes.destNode or nil
                savestates.area.paintingWarpNodes.id[slot] = m.area.paintingWarpNodes.id or nil
            end

            savestates.area.terrainType[slot] = m.area.terrainType or nil

            if m.area.warpNodes then
                savestates.area.warpNodes.next[slot] = m.area.warpNodes.next or nil

                if m.area.warpNodes.node then
                    savestates.area.warpNodes.node.destArea[slot] = m.area.warpNodes.node.destArea or nil
                    savestates.area.warpNodes.node.destLevel[slot] = m.area.warpNodes.node.destLevel or nil
                    savestates.area.warpNodes.node.destNode[slot] = m.area.warpNodes.node.destNode or nil
                    savestates.area.warpNodes.node.id[slot] = m.area.warpNodes.node.id or nil
                end

                savestates.area.warpNodes.object[slot] = m.area.warpNodes.object or nil
            end
        end

        savestates.bounceSquishTimer[slot] = m.bounceSquishTimer or nil
        savestates.bubbleObj[slot] = m.bubbleObj or nil
        savestates.cap[slot] = m.cap or nil
        savestates.capTimer[slot] = m.capTimer or nil
        savestates.ceil[slot] = m.ceil or nil
        savestates.ceilHeight[slot] = m.ceilHeight or nil
        savestates.character[slot] = m.character or nil
        savestates.collidedObjInteractTypes[slot] = m.collidedObjInteractTypes or nil

        if m.controller then
            savestates.controller.buttonDown[slot] = m.controller.buttonDown or nil
            savestates.controller.buttonPressed[slot] = m.controller.buttonPressed or nil
            savestates.controller.extStickX[slot] = m.controller.extStickX or nil
            savestates.controller.extStickY[slot] = m.controller.extStickY or nil
            savestates.controller.port[slot] = m.controller.port or nil
            savestates.controller.rawStickX[slot] = m.controller.rawStickX or nil
            savestates.controller.rawStickY[slot] = m.controller.rawStickY or nil
            savestates.controller.stickMag[slot] = m.controller.stickMag or nil
            savestates.controller.stickX[slot] = m.controller.stickX or nil
            savestates.controller.stickY[slot] = m.controller.stickY or nil
        end

        savestates.curAnimOffset[slot] = m.curAnimOffset or nil
        savestates.currentRoom[slot] = m.currentRoom or nil
        savestates.doubleJumpTimer[slot] = m.doubleJumpTimer or nil

        if m.faceAngle then
            savestates.faceAngle.x[slot] = m.faceAngle.x or nil
            savestates.faceAngle.y[slot] = m.faceAngle.y or nil
            savestates.faceAngle.z[slot] = m.faceAngle.z or nil
        end

        savestates.fadeWarpOpacity[slot] = m.fadeWarpOpacity or nil
        savestates.flags[slot] = m.flags or nil
        savestates.floor[slot] = m.floor or nil
        savestates.floorAngle[slot] = m.floorAngle or nil
        savestates.floorHeight[slot] = m.floorHeight or nil
        savestates.forwardVel[slot] = m.forwardVel or nil
        savestates.framesSinceA[slot] = m.framesSinceA or nil
        savestates.framesSinceB[slot] = m.framesSinceB or nil
        savestates.freeze[slot] = m.freeze or nil
        savestates.healCounter[slot] = m.healCounter or nil
        savestates.health[slot] = m.health or nil
        savestates.heldByObj[slot] = m.heldByObj or nil
        savestates.heldObj[slot] = m.heldObj or nil
        savestates.hurtCounter[slot] = m.hurtCounter or nil
        savestates.input[slot] = m.input or nil
        savestates.intendedMag[slot] = m.intendedMag or nil
        savestates.intendedYaw[slot] = m.intendedYaw or nil
        savestates.interactObj[slot] = m.interactObj or nil
        savestates.invincTimer[slot] = m.invincTimer or nil
        savestates.isSnoring[slot] = m.isSnoring or nil
        savestates.knockbackTimer[slot] = m.knockbackTimer or nil

        if m.marioBodyState then
            savestates.marioBodyState.action[slot] = m.marioBodyState.action or nil
            savestates.marioBodyState.capState[slot] = m.marioBodyState.capState or nil
            savestates.marioBodyState.eyeState[slot] = m.marioBodyState.eyeState or nil
            savestates.marioBodyState.grabPos[slot] = m.marioBodyState.grabPos or nil
            savestates.marioBodyState.handState[slot] = m.marioBodyState.handState or nil

            if m.marioBodyState.headAngle then
                savestates.marioBodyState.headAngle.x[slot] = m.marioBodyState.headAngle.x or nil
                savestates.marioBodyState.headAngle.y[slot] = m.marioBodyState.headAngle.y or nil
                savestates.marioBodyState.headAngle.z[slot] = m.marioBodyState.headAngle.z or nil
            end

            if m.marioBodyState.headPos then
                savestates.marioBodyState.headPos.x[slot] = m.marioBodyState.headPos.x or nil
                savestates.marioBodyState.headPos.y[slot] = m.marioBodyState.headPos.y or nil
                savestates.marioBodyState.headPos.z[slot] = m.marioBodyState.headPos.z or nil
            end

            if m.marioBodyState.heldObjLastPosition then
                savestates.marioBodyState.heldObjLastPosition.x[slot] = m.marioBodyState.heldObjLastPosition.x or nil
                savestates.marioBodyState.heldObjLastPosition.y[slot] = m.marioBodyState.heldObjLastPosition.y or nil
                savestates.marioBodyState.heldObjLastPosition.z[slot] = m.marioBodyState.heldObjLastPosition.z or nil
            end

            savestates.marioBodyState.modelState[slot] = m.marioBodyState.modelState or nil
            savestates.marioBodyState.punchState[slot] = m.marioBodyState.punchState or nil

            if m.marioBodyState.torsoAngle then
                savestates.marioBodyState.torsoAngle.x[slot] = m.marioBodyState.torsoAngle.x or nil
                savestates.marioBodyState.torsoAngle.y[slot] = m.marioBodyState.torsoAngle.y or nil
                savestates.marioBodyState.torsoAngle.z[slot] = m.marioBodyState.torsoAngle.z or nil
            end

            if m.marioBodyState.torsoPos then
                savestates.marioBodyState.torsoPos.x[slot] = m.marioBodyState.torsoPos.x or nil
                savestates.marioBodyState.torsoPos.y[slot] = m.marioBodyState.torsoPos.y or nil
                savestates.marioBodyState.torsoPos.z[slot] = m.marioBodyState.torsoPos.z or nil
            end

            savestates.marioBodyState.wingFlutter[slot] = m.marioBodyState.wingFlutter or nil
        end

        if m.marioObj then
            savestates.marioObj.activeFlags[slot] = m.marioObj.activeFlags or nil
            savestates.marioObj.areaTimer[slot] = m.marioObj.areaTimer or nil
            savestates.marioObj.areaTimerDuration[slot] = m.marioObj.areaTimerDuration or nil
            savestates.marioObj.areaTimerType[slot] = m.marioObj.areaTimerType or nil
            savestates.marioObj.bhvDelayTimer[slot] = m.marioObj.bhvDelayTimer or nil
            savestates.marioObj.collidedObjInteractTypes[slot] = m.marioObj.collidedObjInteractTypes or nil
            savestates.marioObj.collisionData[slot] = m.marioObj.collisionData or nil
            savestates.marioObj.ctx[slot] = m.marioObj.ctx or nil
            savestates.marioObj.globalPlayerIndex[slot] = m.marioObj.globalPlayerIndex or nil

            if m.marioObj.header and m.marioObj.header.gfx then
                savestates.marioObj.header.gfx.activeAreaIndex[slot] = m.marioObj.header.gfx.activeAreaIndex or nil
                savestates.marioObj.header.gfx.areaIndex[slot] = m.marioObj.header.gfx.areaIndex or nil
                savestates.marioObj.header.gfx.disableAutomaticShadowPos[slot] = m.marioObj.header.gfx.disableAutomaticShadowPos or nil
                savestates.marioObj.header.gfx.shadowInvisible[slot] = m.marioObj.header.gfx.shadowInvisible or nil
                savestates.marioObj.header.gfx.skipInViewCheck[slot] = m.marioObj.header.gfx.skipInViewCheck or nil
            end

            savestates.marioObj.heldByPlayerIndex[slot] = m.marioObj.heldByPlayerIndex or nil
            savestates.marioObj.hitboxDownOffset[slot] = m.marioObj.hitboxDownOffset or nil
            savestates.marioObj.hitboxHeight[slot] = m.marioObj.hitboxHeight or nil
            savestates.marioObj.hitboxRadius[slot] = m.marioObj.hitboxRadius or nil
            savestates.marioObj.hookRender[slot] = m.marioObj.hookRender or nil
            savestates.marioObj.hurtboxHeight[slot] = m.marioObj.hurtboxHeight or nil
            savestates.marioObj.hurtboxRadius[slot] = m.marioObj.hurtboxRadius or nil
            savestates.marioObj.parentObj[slot] = m.marioObj.parentObj or nil
            savestates.marioObj.platform[slot] = m.marioObj.platform or nil
            savestates.marioObj.prevObj[slot] = m.marioObj.prevObj or nil
            savestates.marioObj.setHome[slot] = m.marioObj.setHome or nil
            savestates.marioObj.unused1[slot] = m.marioObj.unused1 or nil
            savestates.marioObj.usingObj[slot] = m.marioObj.usingObj or nil
        end

        savestates.minimumBoneY[slot] = m.minimumBoneY or nil

        if m.nonInstantWarpPos then
            savestates.nonInstantWarpPos.x[slot] = m.nonInstantWarpPos.x or nil
            savestates.nonInstantWarpPos.y[slot] = m.nonInstantWarpPos.y or nil
            savestates.nonInstantWarpPos.z[slot] = m.nonInstantWarpPos.z or nil
        end

        savestates.numCoins[slot] = m.numCoins or nil
        savestates.numKeys[slot] = m.numKeys or nil
        savestates.numLives[slot] = m.numLives or nil
        savestates.numStars[slot] = m.numStars or nil
        savestates.particleFlags[slot] = m.particleFlags or nil
        savestates.peakHeight[slot] = m.peakHeight or nil

        if m.pos then
            savestates.pos.x[slot] = m.pos.x or nil
            savestates.pos.y[slot] = m.pos.y or nil
            savestates.pos.z[slot] = m.pos.z or nil
        end

        savestates.prevAction[slot] = m.prevAction or nil
        savestates.prevNumStarsForDialog[slot] = m.prevNumStarsForDialog or nil
        savestates.quicksandDepth[slot] = m.quicksandDepth or nil
        savestates.riddenObj[slot] = m.riddenObj or nil
        savestates.slideVelX[slot] = m.slideVelX or nil
        savestates.slideVelZ[slot] = m.slideVelZ or nil
        savestates.slideYaw[slot] = m.slideYaw or nil
        savestates.spawnInfo[slot] = m.spawnInfo or nil
        savestates.specialTripleJump[slot] = m.specialTripleJump or nil
        savestates.splineKeyframe[slot] = m.splineKeyframe or nil
        savestates.splineKeyframeFraction[slot] = m.splineKeyframeFraction or nil
        savestates.splineState[slot] = m.splineState or nil
        savestates.squishTimer[slot] = m.squishTimer or nil

        if m.statusForCamera then
            savestates.statusForCamera.action[slot] = m.statusForCamera.action or nil
            savestates.statusForCamera.cameraEvent[slot] = m.statusForCamera.cameraEvent or nil

            if m.statusForCamera.faceAngle then
                savestates.statusForCamera.faceAngle.x[slot] = m.statusForCamera.faceAngle.x or nil
                savestates.statusForCamera.faceAngle.y[slot] = m.statusForCamera.faceAngle.y or nil
                savestates.statusForCamera.faceAngle.z[slot] = m.statusForCamera.faceAngle.z or nil
            end

            if m.statusForCamera.headRotation then
                savestates.statusForCamera.headRotation.x[slot] = m.statusForCamera.headRotation.x or nil
                savestates.statusForCamera.headRotation.y[slot] = m.statusForCamera.headRotation.y or nil
                savestates.statusForCamera.headRotation.z[slot] = m.statusForCamera.headRotation.z or nil
            end

            if m.statusForCamera.pos then
                savestates.statusForCamera.pos.x[slot] = m.statusForCamera.pos.x or nil
                savestates.statusForCamera.pos.y[slot] = m.statusForCamera.pos.y or nil
                savestates.statusForCamera.pos.z[slot] = m.statusForCamera.pos.z or nil
            end

            savestates.statusForCamera.unused[slot] = m.statusForCamera.unused or nil
            savestates.statusForCamera.usedObj[slot] = m.statusForCamera.usedObj or nil
        end

        savestates.terrainSoundAddend[slot] = m.terrainSoundAddend or nil
        savestates.twirlYaw[slot] = m.twirlYaw or nil
        savestates.unkB0[slot] = m.unkB0 or nil
        savestates.unkC4[slot] = m.unkC4 or nil
        savestates.usedObj[slot] = m.usedObj or nil

        if m.vel then
            savestates.vel.x[slot] = m.vel.x or nil
            savestates.vel.y[slot] = m.vel.y or nil
            savestates.vel.z[slot] = m.vel.z or nil
        end

        savestates.wall[slot] = m.wall or nil
        savestates.wallKickTimer[slot] = m.wallKickTimer or nil

        if m.wallNormal then
            savestates.wallNormal.x[slot] = m.wallNormal.x or nil
            savestates.wallNormal.y[slot] = m.wallNormal.y or nil
            savestates.wallNormal.z[slot] = m.wallNormal.z or nil
        end

        savestates.wasNetworkVisible[slot] = m.wasNetworkVisible or nil
        savestates.waterLevel[slot] = m.waterLevel or nil
    end

    latest_savestate = slot
    m.particleFlags = PARTICLE_SPARKLES
    play_sound(SOUND_GENERAL_GRAND_STAR, m.marioObj.header.gfx.cameraToObject)
    djui_popup_create(color .. "State saved to Slot " .. tostring(latest_savestate + 1) .. "!", 1)
    return true
end

function load_state_slot(m, slot)
    ready_to_load = nil
    if (is_state_slot_not_empty(slot)) then
        if (savestates.gNetworkPlayers.currLevelNum[slot] ~= gNetworkPlayers[0].currLevelNum or savestates.gNetworkPlayers.currAreaIndex[slot] ~= gNetworkPlayers[0].currAreaIndex or savestates.gNetworkPlayers.currActNum[slot] ~= gNetworkPlayers[0].currActNum) then
            warp_to_level(savestates.gNetworkPlayers.currLevelNum[slot], savestates.gNetworkPlayers.currAreaIndex[slot], savestates.gNetworkPlayers.currActNum[slot])
            wait_until_inited = slot
            return false
        else
			if m then
				if m.action then m.action = savestates.action[slot] end
				if m.actionArg then m.actionArg = savestates.actionArg[slot] end
				if m.actionState then m.actionState = savestates.actionState[slot] end
				if m.actionTimer then m.actionTimer = savestates.actionTimer[slot] end
			
				if m.angleVel then
					if m.angleVel.x then m.angleVel.x = savestates.angleVel.x[slot] end
					if m.angleVel.y then m.angleVel.y = savestates.angleVel.y[slot] end
					if m.angleVel.z then m.angleVel.z = savestates.angleVel.z[slot] end
				end
			
				if m.animation and m.animation.targetAnim then m.animation.targetAnim = savestates.animation.targetAnim[slot] end
			
				if m.area then
					if m.area.camera then m.area.camera = savestates.area.camera[slot] end
					if m.area.flags then m.area.flags = savestates.area.flags[slot] end
					if m.area.index then m.area.index = savestates.area.index[slot] end
					if m.area.instantWarps then m.area.instantWarps = savestates.area.instantWarps[slot] end
					if m.area.musicParam then m.area.musicParam = savestates.area.musicParam[slot] end
					if m.area.musicParam2 then m.area.musicParam2 = savestates.area.musicParam2[slot] end
			
					if m.area.objectSpawnInfos then
						if m.area.objectSpawnInfos.activeAreaIndex then m.area.objectSpawnInfos.activeAreaIndex = savestates.area.objectSpawnInfos.activeAreaIndex[slot] end
						if m.area.objectSpawnInfos.areaIndex then m.area.objectSpawnInfos.areaIndex = savestates.area.objectSpawnInfos.areaIndex[slot] end
						if m.area.objectSpawnInfos.behaviorArg then m.area.objectSpawnInfos.behaviorArg = savestates.area.objectSpawnInfos.behaviorArg[slot] end
			
						if m.area.objectSpawnInfos.startAngle then
							if m.area.objectSpawnInfos.startAngle.x then m.area.objectSpawnInfos.startAngle.x = savestates.area.objectSpawnInfos.startAngle.x[slot] end
							if m.area.objectSpawnInfos.startAngle.y then m.area.objectSpawnInfos.startAngle.y = savestates.area.objectSpawnInfos.startAngle.y[slot] end
							if m.area.objectSpawnInfos.startAngle.z then m.area.objectSpawnInfos.startAngle.z = savestates.area.objectSpawnInfos.startAngle.z[slot] end
						end
			
						if m.area.objectSpawnInfos.startPos then
							if m.area.objectSpawnInfos.startPos.x then m.area.objectSpawnInfos.startPos.x = savestates.area.objectSpawnInfos.startPos.x[slot] end
							if m.area.objectSpawnInfos.startPos.y then m.area.objectSpawnInfos.startPos.y = savestates.area.objectSpawnInfos.startPos.y[slot] end
							if m.area.objectSpawnInfos.startPos.z then m.area.objectSpawnInfos.startPos.z = savestates.area.objectSpawnInfos.startPos.z[slot] end
						end
			
						if m.area.objectSpawnInfos.unk18 then
							if m.area.objectSpawnInfos.unk18.extraFlags then m.area.objectSpawnInfos.unk18.extraFlags = savestates.area.objectSpawnInfos.unk18.extraFlags[slot] end
							if m.area.objectSpawnInfos.unk18.flags then m.area.objectSpawnInfos.unk18.flags = savestates.area.objectSpawnInfos.unk18.flags[slot] end
						end
					end
			
					if m.area.terrainType then m.area.terrainType = savestates.area.terrainType[slot] end
				end
			
				if m.bounceSquishTimer then m.bounceSquishTimer = savestates.bounceSquishTimer[slot] end
				if m.bubbleObj then m.bubbleObj = savestates.bubbleObj[slot] end
				if m.cap then m.cap = savestates.cap[slot] end
				if m.capTimer then m.capTimer = savestates.capTimer[slot] end
				if m.ceil then m.ceil = savestates.ceil[slot] end
				if m.ceilHeight then m.ceilHeight = savestates.ceilHeight[slot] end
				if m.character then m.character = savestates.character[slot] end
				if m.collidedObjInteractTypes then m.collidedObjInteractTypes = savestates.collidedObjInteractTypes[slot] end
				if m.curAnimOffset then m.curAnimOffset = savestates.curAnimOffset[slot] end
				if m.currentRoom then m.currentRoom = savestates.currentRoom[slot] end
				if m.doubleJumpTimer then m.doubleJumpTimer = savestates.doubleJumpTimer[slot] end
			
				if m.faceAngle then
					if m.faceAngle.x then m.faceAngle.x = savestates.faceAngle.x[slot] end
					if m.faceAngle.y then m.faceAngle.y = savestates.faceAngle.y[slot] end
					if m.faceAngle.z then m.faceAngle.z = savestates.faceAngle.z[slot] end
				end
			
				if m.fadeWarpOpacity then m.fadeWarpOpacity = savestates.fadeWarpOpacity[slot] end
				if m.flags then m.flags = savestates.flags[slot] end
				if m.floor then m.floor = savestates.floor[slot] end
				if m.floorAngle then m.floorAngle = savestates.floorAngle[slot] end
				if m.floorHeight then m.floorHeight = savestates.floorHeight[slot] end
				if m.forwardVel then m.forwardVel = savestates.forwardVel[slot] end
				if m.framesSinceA then m.framesSinceA = savestates.framesSinceA[slot] end
				if m.framesSinceB then m.framesSinceB = savestates.framesSinceB[slot] end
				if m.freeze then m.freeze = savestates.freeze[slot] end
				if m.healCounter then m.healCounter = savestates.healCounter[slot] end
				if m.health then m.health = (autoheal_enabled and 0x880 or savestates.health[slot]) end
				if m.heldByObj then m.heldByObj = savestates.heldByObj[slot] end
				if m.heldObj then m.heldObj = savestates.heldObj[slot] end
				if m.hurtCounter then m.hurtCounter = savestates.hurtCounter[slot] end
				if m.input then m.input = savestates.input[slot] end
				if m.intendedMag then m.intendedMag = savestates.intendedMag[slot] end
				if m.intendedYaw then m.intendedYaw = savestates.intendedYaw[slot] end
				if m.interactObj then m.interactObj = savestates.interactObj[slot] end
				if m.invincTimer then m.invincTimer = savestates.invincTimer[slot] end
				if m.isSnoring then m.isSnoring = savestates.isSnoring[slot] end
				if m.knockbackTimer then m.knockbackTimer = savestates.knockbackTimer[slot] end
			
				if m.marioBodyState then
					if m.marioBodyState.action then m.marioBodyState.action = savestates.marioBodyState.action[slot] end
					if m.marioBodyState.capState then m.marioBodyState.capState = savestates.marioBodyState.capState[slot] end
					if m.marioBodyState.eyeState then m.marioBodyState.eyeState = savestates.marioBodyState.eyeState[slot] end
					if m.marioBodyState.grabPos then m.marioBodyState.grabPos = savestates.marioBodyState.grabPos[slot] end
			
					if m.marioBodyState.headAngle then
						if m.marioBodyState.headAngle.x then m.marioBodyState.headAngle.x = savestates.marioBodyState.headAngle.x[slot] end
						if m.marioBodyState.headAngle.y then m.marioBodyState.headAngle.y = savestates.marioBodyState.headAngle.y[slot] end
						if m.marioBodyState.headAngle.z then m.marioBodyState.headAngle.z = savestates.marioBodyState.headAngle.z[slot] end
					end
			
					if m.marioBodyState.headPos then
						if m.marioBodyState.headPos.x then m.marioBodyState.headPos.x = savestates.marioBodyState.headPos.x[slot] end
						if m.marioBodyState.headPos.y then m.marioBodyState.headPos.y = savestates.marioBodyState.headPos.y[slot] end
						if m.marioBodyState.headPos.z then m.marioBodyState.headPos.z = savestates.marioBodyState.headPos.z[slot] end
					end
			
					if m.marioBodyState.heldObjLastPosition then
						if m.marioBodyState.heldObjLastPosition.x then m.marioBodyState.heldObjLastPosition.x = savestates.marioBodyState.heldObjLastPosition.x[slot] end
						if m.marioBodyState.heldObjLastPosition.y then m.marioBodyState.heldObjLastPosition.y = savestates.marioBodyState.heldObjLastPosition.y[slot] end
						if m.marioBodyState.heldObjLastPosition.z then m.marioBodyState.heldObjLastPosition.z = savestates.marioBodyState.heldObjLastPosition.z[slot] end
					end
			
					if m.marioBodyState.modelState then m.marioBodyState.modelState = savestates.marioBodyState.modelState[slot] end
					if m.marioBodyState.punchState then m.marioBodyState.punchState = savestates.marioBodyState.punchState[slot] end
			
					if m.marioBodyState.torsoAngle then
						if m.marioBodyState.torsoAngle.x then m.marioBodyState.torsoAngle.x = savestates.marioBodyState.torsoAngle.x[slot] end
						if m.marioBodyState.torsoAngle.y then m.marioBodyState.torsoAngle.y = savestates.marioBodyState.torsoAngle.y[slot] end
						if m.marioBodyState.torsoAngle.z then m.marioBodyState.torsoAngle.z = savestates.marioBodyState.torsoAngle.z[slot] end
					end
			
					if m.marioBodyState.torsoPos then
						if m.marioBodyState.torsoPos.x then m.marioBodyState.torsoPos.x = savestates.marioBodyState.torsoPos.x[slot] end
						if m.marioBodyState.torsoPos.y then m.marioBodyState.torsoPos.y = savestates.marioBodyState.torsoPos.y[slot] end
						if m.marioBodyState.torsoPos.z then m.marioBodyState.torsoPos.z = savestates.marioBodyState.torsoPos.z[slot] end
					end
			
					if m.marioBodyState.wingFlutter then m.marioBodyState.wingFlutter = savestates.marioBodyState.wingFlutter[slot] end
				end
			
				if m.marioObj then
					if m.marioObj.activeFlags then m.marioObj.activeFlags = savestates.marioObj.activeFlags[slot] end
					if m.marioObj.areaTimer then m.marioObj.areaTimer = savestates.marioObj.areaTimer[slot] end
					if m.marioObj.areaTimerDuration then m.marioObj.areaTimerDuration = savestates.marioObj.areaTimerDuration[slot] end
					if m.marioObj.areaTimerType then m.marioObj.areaTimerType = savestates.marioObj.areaTimerType[slot] end
					if m.marioObj.bhvDelayTimer then m.marioObj.bhvDelayTimer = savestates.marioObj.bhvDelayTimer[slot] end
					if m.marioObj.collidedObjInteractTypes then m.marioObj.collidedObjInteractTypes = savestates.marioObj.collidedObjInteractTypes[slot] end
					if m.marioObj.collisionData then m.marioObj.collisionData = savestates.marioObj.collisionData[slot] end
					if m.marioObj.ctx then m.marioObj.ctx = savestates.marioObj.ctx[slot] end
					if m.marioObj.globalPlayerIndex then m.marioObj.globalPlayerIndex = savestates.marioObj.globalPlayerIndex[slot] end
			
					if m.marioObj.header and m.marioObj.header.gfx then
						if m.marioObj.header.gfx.activeAreaIndex then m.marioObj.header.gfx.activeAreaIndex = savestates.marioObj.header.gfx.activeAreaIndex[slot] end
						if m.marioObj.header.gfx.areaIndex then m.marioObj.header.gfx.areaIndex = savestates.marioObj.header.gfx.areaIndex[slot] end
						if m.marioObj.header.gfx.disableAutomaticShadowPos then m.marioObj.header.gfx.disableAutomaticShadowPos = savestates.marioObj.header.gfx.disableAutomaticShadowPos[slot] end
						if m.marioObj.header.gfx.shadowInvisible then m.marioObj.header.gfx.shadowInvisible = savestates.marioObj.header.gfx.shadowInvisible[slot] end
						if m.marioObj.header.gfx.skipInViewCheck then m.marioObj.header.gfx.skipInViewCheck = savestates.marioObj.header.gfx.skipInViewCheck[slot] end
					end
			
					if m.marioObj.heldByPlayerIndex then m.marioObj.heldByPlayerIndex = savestates.marioObj.heldByPlayerIndex[slot] end
					if m.marioObj.hitboxDownOffset then m.marioObj.hitboxDownOffset = savestates.marioObj.hitboxDownOffset[slot] end
					if m.marioObj.hitboxHeight then m.marioObj.hitboxHeight = savestates.marioObj.hitboxHeight[slot] end
					if m.marioObj.hitboxRadius then m.marioObj.hitboxRadius = savestates.marioObj.hitboxRadius[slot] end
					if m.marioObj.hookRender then m.marioObj.hookRender = savestates.marioObj.hookRender[slot] end
					if m.marioObj.hurtboxHeight then m.marioObj.hurtboxHeight = savestates.marioObj.hurtboxHeight[slot] end
					if m.marioObj.hurtboxRadius then m.marioObj.hurtboxRadius = savestates.marioObj.hurtboxRadius[slot] end
					if m.marioObj.parentObj then m.marioObj.parentObj = savestates.marioObj.parentObj[slot] end
					if m.marioObj.platform then m.marioObj.platform = savestates.marioObj.platform[slot] end
					if m.marioObj.prevObj then m.marioObj.prevObj = savestates.marioObj.prevObj[slot] end
					if m.marioObj.setHome then m.marioObj.setHome = savestates.marioObj.setHome[slot] end
					if m.marioObj.unused1 then m.marioObj.unused1 = savestates.marioObj.unused1[slot] end
					if m.marioObj.usingObj then m.marioObj.usingObj = savestates.marioObj.usingObj[slot] end
				end
			
				if m.minimumBoneY then m.minimumBoneY = savestates.minimumBoneY[slot] end
			
				if m.nonInstantWarpPos then
					if m.nonInstantWarpPos.x then m.nonInstantWarpPos.x = savestates.nonInstantWarpPos.x[slot] end
					if m.nonInstantWarpPos.y then m.nonInstantWarpPos.y = savestates.nonInstantWarpPos.y[slot] end
					if m.nonInstantWarpPos.z then m.nonInstantWarpPos.z = savestates.nonInstantWarpPos.z[slot] end
				end
			
				if m.numLives then m.numLives = savestates.numLives[slot] end
				if m.particleFlags then m.particleFlags = savestates.particleFlags[slot] end
				if m.peakHeight then m.peakHeight = savestates.peakHeight[slot] end
			
				if m.pos then
					if m.pos.x then m.pos.x = savestates.pos.x[slot] end
					if m.pos.y then m.pos.y = savestates.pos.y[slot] end
					if m.pos.z then m.pos.z = savestates.pos.z[slot] end
				end
			
				if m.prevAction then m.prevAction = savestates.prevAction[slot] end
				if m.prevNumStarsForDialog then m.prevNumStarsForDialog = savestates.prevNumStarsForDialog[slot] end
				if m.quicksandDepth then m.quicksandDepth = savestates.quicksandDepth[slot] end
				if m.riddenObj then m.riddenObj = savestates.riddenObj[slot] end
				if m.slideVelX then m.slideVelX = savestates.slideVelX[slot] end
				if m.slideVelZ then m.slideVelZ = savestates.slideVelZ[slot] end
				if m.slideYaw then m.slideYaw = savestates.slideYaw[slot] end
				if m.spawnInfo then m.spawnInfo = savestates.spawnInfo[slot] end
				if m.specialTripleJump then m.specialTripleJump = savestates.specialTripleJump[slot] end
				if m.splineKeyframe then m.splineKeyframe = savestates.splineKeyframe[slot] end
				if m.splineKeyframeFraction then m.splineKeyframeFraction = savestates.splineKeyframeFraction[slot] end
				if m.splineState then m.splineState = savestates.splineState[slot] end
				if m.squishTimer then m.squishTimer = savestates.squishTimer[slot] end
			
				if m.statusForCamera then
					if m.statusForCamera.action then m.statusForCamera.action = savestates.statusForCamera.action[slot] end
					if m.statusForCamera.cameraEvent then m.statusForCamera.cameraEvent = savestates.statusForCamera.cameraEvent[slot] end
			
					if m.statusForCamera.faceAngle then
						if m.statusForCamera.faceAngle.x then m.statusForCamera.faceAngle.x = savestates.statusForCamera.faceAngle.x[slot] end
						if m.statusForCamera.faceAngle.y then m.statusForCamera.faceAngle.y = savestates.statusForCamera.faceAngle.y[slot] end
						if m.statusForCamera.faceAngle.z then m.statusForCamera.faceAngle.z = savestates.statusForCamera.faceAngle.z[slot] end
					end
			
					if m.statusForCamera.headRotation then
						if m.statusForCamera.headRotation.x then m.statusForCamera.headRotation.x = savestates.statusForCamera.headRotation.x[slot] end
						if m.statusForCamera.headRotation.y then m.statusForCamera.headRotation.y = savestates.statusForCamera.headRotation.y[slot] end
						if m.statusForCamera.headRotation.z then m.statusForCamera.headRotation.z = savestates.statusForCamera.headRotation.z[slot] end
					end
			
					if m.statusForCamera.pos then
						if m.statusForCamera.pos.x then m.statusForCamera.pos.x = savestates.statusForCamera.pos.x[slot] end
						if m.statusForCamera.pos.y then m.statusForCamera.pos.y = savestates.statusForCamera.pos.y[slot] end
						if m.statusForCamera.pos.z then m.statusForCamera.pos.z = savestates.statusForCamera.pos.z[slot] end
					end
			
					if m.statusForCamera.unused then m.statusForCamera.unused = savestates.statusForCamera.unused[slot] end
					if m.statusForCamera.usedObj then m.statusForCamera.usedObj = savestates.statusForCamera.usedObj[slot] end
				end
			
				if m.terrainSoundAddend then m.terrainSoundAddend = savestates.terrainSoundAddend[slot] end
				if m.twirlYaw then m.twirlYaw = savestates.twirlYaw[slot] end
				if m.unkB0 then m.unkB0 = savestates.unkB0[slot] end
				if m.unkC4 then m.unkC4 = savestates.unkC4[slot] end
				if m.usedObj then m.usedObj = savestates.usedObj[slot] end
			
				if m.vel then
					if m.vel.x then m.vel.x = savestates.vel.x[slot] end
					if m.vel.y then m.vel.y = savestates.vel.y[slot] end
					if m.vel.z then m.vel.z = savestates.vel.z[slot] end
				end
			
				if m.wall then m.wall = savestates.wall[slot] end
				if m.wallKickTimer then m.wallKickTimer = savestates.wallKickTimer[slot] end
			
				if m.wallNormal then
					if m.wallNormal.x then m.wallNormal.x = savestates.wallNormal.x[slot] end
					if m.wallNormal.y then m.wallNormal.y = savestates.wallNormal.y[slot] end
					if m.wallNormal.z then m.wallNormal.z = savestates.wallNormal.z[slot] end
				end
			
				if m.wasNetworkVisible then m.wasNetworkVisible = savestates.wasNetworkVisible[slot] end
				if m.waterLevel then m.waterLevel = savestates.waterLevel[slot] end
			end
			
            latest_savestate = slot
            m.particleFlags = PARTICLE_HORIZONTAL_STAR
            play_sound(SOUND_GENERAL_GRAND_STAR_JUMP, m.marioObj.header.gfx.cameraToObject)
            if slot == 0 then
                djui_popup_create("\\#ffa0a0\\State loaded from Slot 1!",1)
            elseif slot == 1 then
                djui_popup_create("\\#a0ffa0\\State loaded from Slot 2!",1)
            elseif slot == 2 then
                djui_popup_create("\\#a0a0ff\\State loaded from Slot 3!",1)
            elseif slot == 3 then
                djui_popup_create("\\#ffffa0\\State loaded from Slot 4!",1)
            end
            return true
        end
    else
        play_sound(SOUND_GENERAL_FLAME_OUT, m.marioObj.header.gfx.cameraToObject)
        if slot == 0 then
            djui_popup_create("\\#a0a0a0\\No State found in Slot 1!",1)
        elseif slot == 1 then
            djui_popup_create("\\#a0a0a0\\No State found in Slot 2!",1)
        elseif slot == 2 then
            djui_popup_create("\\#a0a0a0\\No State found in Slot 3!",1)
        elseif slot == 3 then
            djui_popup_create("\\#a0a0a0\\No State found in Slot 4!",1)
        end
        return false
    end
end

function delete_state_slot(m, slot)
    savestates.pos.x[slot] = nil
    savestates.pos.y[slot] = nil
    savestates.pos.z[slot] = nil
end

function is_state_slot_not_empty(slot) 
    return ((savestates.pos.x[slot] ~= nil and savestates.pos.y[slot] ~= nil and savestates.pos.z[slot] ~= nil) and true or false)
end

function on_cmd_autoload()
    if (autoload_enabled == true) then
        autoload_enabled = false
        djui_chat_message_create("\\#ff8080\\AutoLoad latest savestate on death is now \\#ff0000\\DISABLED")
    else
        autoload_enabled = true
        djui_chat_message_create("\\#80ff80\\AutoLoad latest savestate on death is now \\#00ff00\\ENABLED")
    end
    return true
end

function on_cmd_autoheal()
    if (autoheal_enabled == true) then
        autoheal_enabled = false
        djui_chat_message_create("\\#ff8080\\AutoHeal after loading savestate is now \\#ff0000\\DISABLED")
    else
        autoheal_enabled = true
        djui_chat_message_create("\\#80ff80\\AutoHeal after loading savestate is now \\#00ff00\\ENABLED")
    end
    return true
end

function on_cmd_savestate()
    djui_chat_message_create("\\#40b0ff\\[\\#40ffff\\LOCAL SAVESTATE-MOD SLOTS AND SETTINGS INFOS\\#40b0ff\\]")
    djui_chat_message_create("\\#b0b0b0\\AutoLoad\\#808080\\: " .. (autoload_enabled and "\\#80ff80\\ON" or "\\#ff8080\\OFF") .. " " .. (is_state_slot_not_empty(latest_savestate) and ("\\#ffff80\\(Latest: Slot " .. tostring(latest_savestate + 1) .. ")") or ("\\#ff8040\\(Latest: MissingNo)")) .. "    " .. "\\#b0b0b0\\AutoHeal\\#808080\\: " .. (autoheal_enabled and "\\#80ff80\\ON" or "\\#ff8080\\OFF"))
    djui_chat_message_create("\\#b0b0b0\\Save-Slot 1\\#808080\\: " .. (is_state_slot_not_empty(0) and "\\#80ff80\\SET" or "\\#ff8080\\EMPTY") .. "    " .. "\\#b0b0b0\\Save-Slot 2\\#808080\\: " .. (is_state_slot_not_empty(1) and "\\#80ff80\\SET" or "\\#ff8080\\EMPTY"))
    djui_chat_message_create("\\#b0b0b0\\Save-Slot 3\\#808080\\: " .. (is_state_slot_not_empty(2) and "\\#80ff80\\SET" or "\\#ff8080\\EMPTY") .. "    " .. "\\#b0b0b0\\Save-Slot 4\\#808080\\: " .. (is_state_slot_not_empty(3) and "\\#80ff80\\SET" or "\\#ff8080\\EMPTY"))
    return true
end

hook_event(HOOK_ON_PLAYER_CONNECTED, on_join)
hook_event(HOOK_ON_PLAYER_DISCONNECTED, on_leave)
hook_event(HOOK_ON_DEATH, on_death)
hook_event(HOOK_ON_LEVEL_INIT, on_level_init)
hook_event(HOOK_MARIO_UPDATE, on_mario_update)

hook_chat_command("autoload", "Automatically load the latest savestate on death!", on_cmd_autoload)
hook_chat_command("al", "Automatically load the latest savestate on death!", on_cmd_autoload)

hook_chat_command("autoheal", "Automatically heal mario after loading a savestates!", on_cmd_autoheal)
hook_chat_command("ah", "Automatically heal mario after loading a savestates!", on_cmd_autoheal)

hook_chat_command("savestate", "List infos about your local savestates and settings!", on_cmd_savestate)
hook_chat_command("savestates", "List infos about your local savestates and settings!", on_cmd_savestate)
hook_chat_command("ss", "List infos about your local savestates and settings!", on_cmd_savestate)

-- name: AI mod
-- description: poop
local tagCompatibilityMode = false
local lockCameraMode = true

if not tagCompatibilityMode then
    gLevelValues.entryLevel = LEVEL_BOB
    -- gLevelValues.entryLevel = LEVEL_BITDW
    gServerSettings.skipIntro = 1
    gServerSettings.nametags = 0
end


local my_start_pos = {x = 0, y = 0, z = 0}
local last_angle = 0
local first_angle = 0
local frameCounted = 0



function on_init()
    vec3f_set(my_start_pos, gMarioStates[0].pos.x, gMarioStates[0].pos.y, gMarioStates[0].pos.z)
    first_angle = gMarioStates[0].faceAngle.y

    -- set_override_fov(0.0001)
    set_override_fov(90)
    
    -- if not network_is_server() then
    --     return
    -- end
    if lockCameraMode then
        camera_freeze()
    end
end

function player_respawn(m)
    -- reset most variables
    init_single_mario(m)
    last_angle = first_angle

    -- spawn location/angle
    m.pos.x = my_start_pos.x
    m.pos.y = my_start_pos.y
    m.pos.z = my_start_pos.z
    m.faceAngle.y = 0
    
    -- reset the rest of the variables
    m.capTimer = 0
    m.health = 0x880
    m.numLives = 6
    soft_reset_camera(m.area.camera)
    stop_cap_music()
end

-- death function
function on_death(m)
    if m.playerIndex ~= 0 then return false end
    player_respawn(m)
    -- print("died\n")
    return false
end


function set_camera(m)
    vec3f_set(gLakituState.focus, m.pos.x, m.pos.y, m.pos.z)

    local faceAngle = last_angle
    if m.vel.z * m.vel.z + m.vel.x * m.vel.x > 100 then
        faceAngle = atan2s(m.vel.z, m.vel.x)
        last_angle = faceAngle
    end
    vec3f_set_dist_and_angle(m.pos, gLakituState.pos, 500, 0, faceAngle + 0x8000)
    gLakituState.pos.y = gLakituState.pos.y + 300
end

function update_mario(m)
    if m.playerIndex == 0 and gNetworkPlayers[0].name == "AI_BOT" then
        -- respawn if L is pressed
        if (m.controller.buttonDown & L_TRIG) ~= 0 then
            player_respawn(m)
        end
        if lockCameraMode then
            set_camera(m)
            -- adding relative controls for locked camera
            if m.controller.stickMag ~= 0 then
                
                m.controller.stickMag = math.sqrt(m.controller.stickY * m.controller.stickY + m.controller.stickX * m.controller.stickX)
                local angle = atan2s(m.controller.stickX, m.controller.stickY) + last_angle + 0x8000
                m.controller.stickX = m.controller.stickMag * coss(angle)
                m.controller.stickY = m.controller.stickMag * sins(angle)
            end
        end
        -- Removing the relative controls
        -- if m.controller.stickMag ~= 0 then
        --     m.controller.stickMag = math.sqrt(m.controller.stickY * m.controller.stickY + m.controller.stickX * m.controller.stickX)
        --     local angle = atan2s(m.controller.stickX, m.controller.stickY) - gLakituState.yaw
        --     m.controller.stickX = m.controller.stickMag * coss(angle)
        --     m.controller.stickY = m.controller.stickMag * sins(angle)
        -- end
    end
end

-- hook_event(HOOK_ON_HUD_RENDER, hud_hide)

hook_event(HOOK_BEFORE_MARIO_UPDATE, update_mario)
hook_event(HOOK_ON_LEVEL_INIT, on_init)
hook_event(HOOK_ON_DIALOG, function () return false end)
hook_event(HOOK_USE_ACT_SELECT, function () return false end)

if not compatibilityMode then
    hook_event(HOOK_ON_DEATH, on_death)
end





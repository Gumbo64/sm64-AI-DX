-- name: \\#B0B000\\Fast Hide and Seek
-- incompatible: gamemode
-- description: Modification of the Hide and Seek gamemode to trap everyone within the same level. \nLevels automatically cycle similarly to the Flood gamemode. \n\nWritten by Dan \n\nHide and Seek mod by Super Keeberghrh. \n\nForced Warps mod by Sunk. \n\nRemove Star Spawn Cutscenes by Sunk. \n\nAutomatic Doors by Sunk and Blocky. \n\nScoreboard regex and colored names by Birdekek and EmeraldLoc. \n\nCode for "always have cap" by Mechstreme. \n\nUnlock/Lock cannons code by EmeraldLockdown \n\nNOTICE \nScript edits are allowed! Anyone can customize the script how they see fit. No credit or permission necessary.

-------------------
-- GAME SETTINGS --
-------------------

-- TIME
local sRoundEndTimeout    = 120 * 30     -- ROUND TIME
local sRoundAddTime       = 15 * 30      -- ADD TIME FOR EACH CATCH/JOIN
local sRoundStartTimeout  = 25 * 30      -- INTERMISSION TIME
local sRoundEndTransition = 10 * 30      -- WARP DURING INTERMISSION

-- STANDARD LEVELS
-- The default level set. This is to filter out unwanted levels.
local standardLevels = {
    
    {LEVEL_CASTLE, 1, 0},
    {LEVEL_BOB, 1, 2},    
    {LEVEL_WF, 1, 2},
    {LEVEL_CCM, 1, 1},
    {LEVEL_BBH, 1, 1},
    {LEVEL_SSL, 1, 1},
    {LEVEL_SL, 1, 1},
    {LEVEL_WDW, 1, 1},
    {LEVEL_TTM, 1, 1},
    {LEVEL_THI, 1, 1},

}

-- ALL LEVELS
-- An optional version to include all 30 areas in the game.
local allLevels = {

    {LEVEL_CASTLE_COURTYARD, 1, 0},
    {LEVEL_BOB, 1, 2},    
    {LEVEL_WF, 1, 2},
    {LEVEL_JRB, 1, 2},
    {LEVEL_CCM, 1, 1},
    {LEVEL_BBH, 1, 1},

    {LEVEL_PSS, 1, 1},
    {LEVEL_SA, 1, 1},
    {LEVEL_TOTWC, 1, 1},
    {LEVEL_BITDW, 1, 1},
    {LEVEL_BOWSER_1, 1, 1},

    {LEVEL_CASTLE, 1, 0},
    {LEVEL_HMC, 1, 1},
    {LEVEL_LLL, 1, 1},
    {LEVEL_SSL, 1, 1},
    {LEVEL_DDD, 1, 0},

    {LEVEL_VCUTM, 1, 1},
    {LEVEL_COTMC, 1, 1},
    {LEVEL_BITFS, 1, 1},
    {LEVEL_BOWSER_2, 1, 1},

    {LEVEL_CASTLE_GROUNDS, 1, 0},
    {LEVEL_SL, 1, 1},
    {LEVEL_WDW, 1, 1},
    {LEVEL_TTM, 1, 1},
    {LEVEL_THI, 1, 1},
    {LEVEL_TTC, 1, 1},
    {LEVEL_RR, 1, 1},

    {LEVEL_WMOTR, 1, 1},
    {LEVEL_BITS, 1, 1},
    {LEVEL_BOWSER_3, 1, 1}

}

-- constants
local ROUND_STATE_WAIT        = 0
local ROUND_STATE_ACTIVE      = 1
local ROUND_STATE_SEEKERS_WIN = 2
local ROUND_STATE_HIDERS_WIN  = 3
local ROUND_STATE_UNKNOWN_END = 4

-- server settings
gServerSettings.bubbleDeath = 0
gServerSettings.stayInLevelAfterStar = 2
gServerSettings.playerKnockbackStrength = 10
gLevelValues.disableActs = true

-- globals
gGlobalSyncTable.roundState   = ROUND_STATE_WAIT -- current round state
gGlobalSyncTable.touchTag = true
gGlobalSyncTable.displayTimer = 0 -- the displayed timer
gGlobalSyncTable.resetStore = 0
gGlobalSyncTable.paused = 1
gGlobalSyncTable.banKoopaShell = true
gGlobalSyncTable.disableBLJ = true
gGlobalSyncTable.rankToggle = false

-- variables
local sRoundTimer         = 0            -- the server's round timer
local pauseExitTimer = 0
local canLeave = false
local sFlashingIndex = 0
local puX = 0
local puZ = 0
local np = gNetworkPlayers[0]
local cannonTimer = 0
local connectedCount = 0

-- force level variables
gGlobalSyncTable.levelList = 1
gGlobalSyncTable.levelIndex = 1
gGlobalSyncTable.levelChangeStore = 0
gGlobalSyncTable.forcedLevel = standardLevels[1][1]
gGlobalSyncTable.forcedArea = standardLevels[1][2]
gGlobalSyncTable.forcedAct = standardLevels[1][3]

-- player scores

for i = 0, MAX_PLAYERS - 1 do
    gPlayerSyncTable[i].hiderScore = 0
end
for i = 0, MAX_PLAYERS - 1 do
    gPlayerSyncTable[i].seekerScore = 0
end



--localize functions to improve performance
local
hook_chat_command, network_player_set_description, hook_on_sync_table_change, network_is_server,
hook_event, djui_popup_create, network_get_player_text_color_string, play_sound,
play_character_sound, djui_chat_message_create, djui_hud_set_resolution, djui_hud_set_font,
djui_hud_set_color, djui_hud_render_rect, djui_hud_print_text, djui_hud_get_screen_width, djui_hud_get_screen_height,
djui_hud_measure_text, tostring, warp_to_level, warp_to_start_level, stop_cap_music, dist_between_objects,
math_floor, math_ceil, table_insert, set_camera_mode, hud_render_power_meter
=
hook_chat_command, network_player_set_description, hook_on_sync_table_change, network_is_server,
hook_event, djui_popup_create, network_get_player_text_color_string, play_sound,
play_character_sound, djui_chat_message_create, djui_hud_set_resolution, djui_hud_set_font,
djui_hud_set_color, djui_hud_render_rect, djui_hud_print_text, djui_hud_get_screen_width, djui_hud_get_screen_height,
djui_hud_measure_text, tostring, warp_to_level, warp_to_start_level, stop_cap_music, dist_between_objects,
math.floor, math.ceil, table.insert, set_camera_mode, hud_render_power_meter

local function on_or_off(value)
    if value then return "enabled" end
    return "disabled"
end

local function server_update()
    -- increment timer
    sRoundTimer = sRoundTimer + 1
    gGlobalSyncTable.displayTimer = math_floor(sRoundTimer / 30)

    -- figure out state of the game
    local hasSeeker = false
    local hasHider = false
    local activePlayers = {}
    connectedCount = 0
    
    for i = 0, (MAX_PLAYERS-1) do
        if gNetworkPlayers[i].connected then
            connectedCount = connectedCount + 1
            table_insert(activePlayers, gPlayerSyncTable[i])
            if gPlayerSyncTable[i].seeking then
                hasSeeker = true
            else
                hasHider = true
            end
        else
            gPlayerSyncTable[i].hiderScore = 0
            gPlayerSyncTable[i].seekerScore = 0
        end
    end

    -- only change state if there are 2+ players
    if connectedCount < 2 then
        gGlobalSyncTable.roundState = ROUND_STATE_WAIT
        return
    elseif gGlobalSyncTable.roundState == ROUND_STATE_WAIT then
        gGlobalSyncTable.roundState = ROUND_STATE_UNKNOWN_END
        sRoundTimer = 0
        gGlobalSyncTable.displayTimer = 0
    end

    -- check to see if the round should end
    if gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
        if not hasHider or not hasSeeker or sRoundTimer > sRoundEndTimeout then
            if gGlobalSyncTable.resetStore == 1 then
                gGlobalSyncTable.roundState = ROUND_STATE_FORCED_END
            elseif not hasHider then
                gGlobalSyncTable.roundState = ROUND_STATE_SEEKERS_WIN
            elseif sRoundTimer > sRoundEndTimeout then
                gGlobalSyncTable.roundState = ROUND_STATE_HIDERS_WIN
                local failedSeekers = 0
                for i = 0, (MAX_PLAYERS-1) do
                    if gNetworkPlayers[i].connected then
                        if gPlayerSyncTable[i].seeking then
                            failedSeekers = failedSeekers + 1
                        end
                    end
                end
                for i = 0, (MAX_PLAYERS-1) do
                    if gNetworkPlayers[i].connected then
                        if not gPlayerSyncTable[i].seeking then
                            gPlayerSyncTable[i].hiderScore = gPlayerSyncTable[i].hiderScore + (failedSeekers*2)
                        end
                    end
                end
            else
                gGlobalSyncTable.roundState = ROUND_STATE_UNKNOWN_END
            end
            sRoundTimer = 0
            gGlobalSyncTable.displayTimer = 0
            if gGlobalSyncTable.resetStore == 1 then
                gGlobalSyncTable.resetStore = 0
            else
                gGlobalSyncTable.levelChangeStore = 1
            end
        else
            return
        end
    end

    --CHANGE LEVEL AFTER ROUND
    if sRoundTimer >= sRoundEndTransition and gGlobalSyncTable.levelChangeStore == 1 then
        
        if gGlobalSyncTable.levelList == 1 then
            if gGlobalSyncTable.levelIndex >= #(standardLevels) then
                gGlobalSyncTable.levelIndex = 1
            else
                gGlobalSyncTable.levelIndex = gGlobalSyncTable.levelIndex + 1
            end
            gGlobalSyncTable.forcedLevel = standardLevels[gGlobalSyncTable.levelIndex][1]
            gGlobalSyncTable.forcedArea = standardLevels[gGlobalSyncTable.levelIndex][2]
            gGlobalSyncTable.forcedAct = standardLevels[gGlobalSyncTable.levelIndex][3]
        else
            if gGlobalSyncTable.levelIndex >= #(allLevels) then
                gGlobalSyncTable.levelIndex = 1
            else
                gGlobalSyncTable.levelIndex = gGlobalSyncTable.levelIndex + 1
            end
            gGlobalSyncTable.forcedLevel = allLevels[gGlobalSyncTable.levelIndex][1]
            gGlobalSyncTable.forcedArea = allLevels[gGlobalSyncTable.levelIndex][2]
            gGlobalSyncTable.forcedAct = allLevels[gGlobalSyncTable.levelIndex][3]
        end

        gGlobalSyncTable.levelChangeStore = 0

    end

    -- start round
    if sRoundTimer >= sRoundStartTimeout then
        -- reset seekers
        for i=0,(MAX_PLAYERS-1) do
            gPlayerSyncTable[i].seeking = false
        end
        hasSeeker = false

        -- pick random seeker
        if not hasSeeker then
            local randNum = math.random(#activePlayers)
            local s = activePlayers[randNum]
            s.seeking = true
            if connectedCount > 4 then
                while s.seeking == true do
                    randNum = math.random(#activePlayers)
                    s = activePlayers[randNum]
                end
                s.seeking = true
            end
            if connectedCount > 10 then
                while s.seeking == true do
                    randNum = math.random(#activePlayers)
                    s = activePlayers[randNum]
                end
                s.seeking = true
            end
        end

        -- set round state
        gGlobalSyncTable.roundState = ROUND_STATE_ACTIVE
        sRoundTimer = 0
        gGlobalSyncTable.displayTimer = 0

    end
end

local function update()

    pauseExitTimer = pauseExitTimer + 1

    if pauseExitTimer >= 900 and not canLeave then
        canLeave = true
    end
    -- only allow the server to figure out the seeker
    if network_is_server() then
        server_update()
    end

    -- ENFORCE STAY IN SAME LEVEL
    if gNetworkPlayers[0].currLevelNum ~= gGlobalSyncTable.forcedLevel then
        warp_to_level(gGlobalSyncTable.forcedLevel, gGlobalSyncTable.forcedArea, gGlobalSyncTable.forcedAct)
        gMarioStates[0].health = 0x800
    elseif gNetworkPlayers[0].currActNum ~= gGlobalSyncTable.forcedAct then
        warp_to_level(gGlobalSyncTable.forcedLevel, gGlobalSyncTable.forcedArea, gGlobalSyncTable.forcedAct)
        gMarioStates[0].health = 0x800
    end
end

-- If a player dies, they become a seeker.
local function screen_transition(trans)
    local s = gPlayerSyncTable[0]
    local m = gMarioStates[0]
    if not s.seeking and gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
        if trans == WARP_TRANSITION_FADE_INTO_BOWSER or (m.floor.type == SURFACE_DEATH_PLANE and m.pos.y <= m.floorHeight + 2048) then
            s.seeking = true
        end
    end
end

---@param m MarioState
local function before_mario_update(m)
    if m.playerIndex ~= 0 then return end
    m.flags = m.flags | MARIO_CAP_ON_HEAD
end

--- @param m MarioState
local function mario_update(m)
    if (m.flags & MARIO_VANISH_CAP) ~= 0 then
        m.flags = m.flags & ~MARIO_VANISH_CAP --Always Remove Vanish Cap
        stop_cap_music()
    end
    if (m.flags & MARIO_WING_CAP) ~= 0 and (gGlobalSyncTable.levelList == 1) then
        m.flags = m.flags & ~MARIO_WING_CAP --Always Remove Wing Cap for Standard Levels
        stop_cap_music()
    end
    if (m.flags & MARIO_METAL_CAP) ~= 0 then
        m.flags = m.flags & ~MARIO_METAL_CAP --Always Remove Metal Cap
        stop_cap_music()
    end

    if gGlobalSyncTable.disableBLJ and m.forwardVel <= -55 then
        m.forwardVel = -55
    end

    -- this code runs for all players
    local s = gPlayerSyncTable[m.playerIndex]

    if m.playerIndex == 0 and m.action == ACT_IN_CANNON and m.actionState == 2 then
        cannonTimer = cannonTimer + 1
        if cannonTimer >= 90 then -- 90 is 3 seconds
            m.forwardVel = 100 * coss(m.faceAngle.x)

            m.vel.y = 100 * sins(m.faceAngle.x)

            m.pos.x = m.pos.x + 120 * coss(m.faceAngle.x) * sins(m.faceAngle.y)
            m.pos.y = m.pos.y + 120 * sins(m.faceAngle.x)
            m.pos.z = m.pos.z + 120 * coss(m.faceAngle.x) * coss(m.faceAngle.y)

            play_sound(SOUND_ACTION_FLYING_FAST, m.marioObj.header.gfx.cameraToObject)
            play_sound(SOUND_OBJ_POUNDING_CANNON, m.marioObj.header.gfx.cameraToObject)

            m.marioObj.header.gfx.node.flags = m.marioObj.header.gfx.node.flags | GRAPH_RENDER_ACTIVE
            set_camera_mode(m.area.camera, m.area.camera.defMode, 1)

            set_mario_action(m, ACT_SHOT_FROM_CANNON, 0)
            queue_rumble_data_mario(m, 60, 70)
            m.usedObj.oAction = 2
            cannonTimer = 0
        end
    end

    if m.playerIndex == 0 and m.action == ACT_SHOT_FROM_CANNON then
        cannonTimer = 0
    end

    -- warp to the beninging
    if m.playerIndex == 0 then
        if gPlayerSyncTable[m.playerIndex].seeking and gGlobalSyncTable.displayTimer == 0 and gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
            warp_restart_level()
        end
    end

    -- display all seekers as metal
    if s.seeking then
        m.marioBodyState.modelState = m.marioBodyState.modelState | MODEL_STATE_METAL
        np.overridePaletteIndex = 24
    else
        np.overridePaletteIndex = np.paletteIndex
    end

    -- pu prevention
    if m.pos.x >= 0 then
        puX = math_floor((8192 + m.pos.x) / 65536)
    else
        puX = math_ceil((-8192 + m.pos.x) / 65536)
    end
    if m.pos.z >= 0 then
        puZ = math_floor((8192 + m.pos.z) / 65536)
    else
        puZ = math_ceil((-8192 + m.pos.z) / 65536)
    end
    if puX ~= 0 or puZ ~= 0 then
        s.seeking = true
        warp_restart_level()
    end

end

---@param m MarioState
---@param action integer
local function before_set_mario_action(m, action)
    if m.playerIndex == 0 then
        if action == ACT_WAITING_FOR_DIALOG or action == ACT_READING_SIGN or action == ACT_READING_NPC_DIALOG or action == ACT_JUMBO_STAR_CUTSCENE then
            return 1
        elseif action == ACT_READING_AUTOMATIC_DIALOG and get_id_from_behavior(m.interactObj.behavior) ~= id_bhvDoor and get_id_from_behavior(m.interactObj.behavior) ~= id_bhvStarDoor then
            return 1
        elseif action == ACT_EXIT_LAND_SAVE_DIALOG then
            set_camera_mode(m.area.camera, m.area.camera.defMode, 1)
            return ACT_IDLE
        end
    end
end

local function on_pvp_attack(attacker, victim)
    -- this code runs when a player attacks another player
    local sAttacker = gPlayerSyncTable[attacker.playerIndex]
    local sVictim = gPlayerSyncTable[victim.playerIndex]

    -- only consider local player
    if victim.playerIndex ~= 0 then
        return
    end

    -- make victim a seeker
    if sAttacker.seeking and not sVictim.seeking then
        
        local seekerCount = 0
        for i = 0, (MAX_PLAYERS-1) do
            if gNetworkPlayers[i].connected then
                if gPlayerSyncTable[i].seeking then
                    seekerCount = seekerCount + 1
                end
            end
        end
        sAttacker.seekerScore = sAttacker.seekerScore + seekerCount
        sVictim.hiderScore = sVictim.hiderScore + seekerCount
        sVictim.seeking = true
        
    end
end

--- @param m MarioState
local function on_player_connected(m)
    -- start out as a seeker
    local s = gPlayerSyncTable[m.playerIndex]
    s.seeking = true
    network_player_set_description(gNetworkPlayers[m.playerIndex], "Seeker", 255, 128, 128, 255)
end

local function hud_top_render()
    local seconds = 0
    local text = ""

    if gGlobalSyncTable.roundState == ROUND_STATE_WAIT then
        seconds = 60
        text = "Waiting for Players..."
    elseif gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
        seconds = math_floor(sRoundEndTimeout / 30 - gGlobalSyncTable.displayTimer)
        if seconds < 0 then seconds = 0 end
        text = "Time: " .. seconds
    else
        seconds = math_floor(sRoundStartTimeout / 30 - gGlobalSyncTable.displayTimer)
        if seconds < 0 then seconds = 0 end
        text = "Next Round: " .. seconds
    end

    local scale = 0.5

    -- get width of screen and text
    local screenWidth = djui_hud_get_screen_width()
    local width = djui_hud_measure_text(text) * scale

    local x = (screenWidth - width) * 0.5
    local y = 0

    local background = 0.0
    if seconds < 60 and gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
        background = (math.sin(sFlashingIndex * 0.1) * 0.5 + 0.5) * 1
        background = background * background
        background = background * background
    end

    -- render top
    djui_hud_set_color(255 * background, 0, 0, 128)
    djui_hud_render_rect(x - 6, y, width + 12, 16)

    djui_hud_set_color(255, 255, 255, 255)
    djui_hud_print_text(text, x, y, scale)
end

local function hud_center_render()
    if gGlobalSyncTable.displayTimer > 3 then return end



    -- set text
    local text = ""
    if gGlobalSyncTable.roundState == ROUND_STATE_SEEKERS_WIN then
        text = "All Hiders Caught!"
    elseif gGlobalSyncTable.roundState == ROUND_STATE_HIDERS_WIN then
        local hiderCount = 0
        for i = 0, MAX_PLAYERS - 1 do 
            if gNetworkPlayers[i].connected and not gPlayerSyncTable[i].seeking then
                hiderCount = hiderCount + 1
            end
        end
        if hiderCount == 1 then
            text = "Last Hider Escaped!"
        else
            text = hiderCount .. " Hiders Escaped!"
        end
        
    elseif gGlobalSyncTable.roundState == ROUND_STATE_FORCED_END then
        text = "Resetting Round"
    elseif gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
        text = "Go!"
    else
        return
    end

    -- set scale
    local scale = 1

    -- get width of screen and text
    local screenWidth = djui_hud_get_screen_width()
    local screenHeight = djui_hud_get_screen_height()
    local width = djui_hud_measure_text(text) * scale
    local height = 32 * scale

    local x = (screenWidth - width) * 0.5
    local y = (screenHeight - height) * 0.5

    -- render
    djui_hud_set_color(0, 0, 0, 192)
    djui_hud_render_rect(x - 6 * scale, y, width + 12 * scale, height)

    djui_hud_set_color(255, 255, 255, 255)
    djui_hud_print_text(text, x, y, scale)
end

local function on_hud_render()
    -- render to N64 screen space, with the HUD font
    djui_hud_set_resolution(RESOLUTION_N64)
    djui_hud_set_font(FONT_NORMAL)

    hud_hide()
    hud_render_power_meter(gMarioStates[0].health, djui_hud_get_screen_width() - 64, 0, 64, 64)

    hud_top_render()
    hud_center_render()

    djui_hud_set_resolution(RESOLUTION_DJUI)
    djui_hud_set_color(0, 0, 0, 192)

    local connCount = 0
    for i = 0, MAX_PLAYERS - 1 do 
        if gNetworkPlayers[i].connected then
            connCount = connCount + 1
        end
    end

    local seekerCount = 0
    for i = 0, MAX_PLAYERS - 1 do 
        if gNetworkPlayers[i].connected and gPlayerSyncTable[i].seeking then
            seekerCount = seekerCount + 1
        end
    end
    
    if gGlobalSyncTable.roundState ~= ROUND_STATE_ACTIVE and gGlobalSyncTable.rankToggle  then

        djui_hud_render_rect(0, 0, 450, (connCount+3) * 30)
        
        djui_hud_set_color(255, 255, 255, 255)
        if gGlobalSyncTable.levelList == 1 then
            djui_hud_print_text("Round " .. gGlobalSyncTable.levelIndex .. "/" .. #(standardLevels), 4, 0, 1)
        else
            djui_hud_print_text("Round " .. gGlobalSyncTable.levelIndex .. "/" .. #(allLevels), 4, 0, 1)
        end
        
        djui_hud_print_text("Players (" .. connCount .. ")", 4, 60, 1)
        djui_hud_print_text("Rating", 180, 60, 1)
        djui_hud_print_text("H-Score", 270, 60, 1)
        djui_hud_print_text("S-Score", 360, 60, 1)



        local players = {}

        for i = 0, MAX_PLAYERS - 1 do
            local np = gNetworkPlayers[i]
            if np.connected then
                table.insert(players, i)
            end
        end

        table.sort(players, function (a, b)
            return (gPlayerSyncTable[a].hiderScore + gPlayerSyncTable[a].seekerScore) > (gPlayerSyncTable[b].hiderScore + gPlayerSyncTable[b].seekerScore)
        end)

        local j = 2

        for s = 1, #players do 
            local i = players[s]
            if gNetworkPlayers[i].connected then
                
                j = j + 1

                local r, g, b = hex_to_rgb(network_get_player_text_color_string(i))
                djui_hud_set_color(r, g, b, 255)

                local name = string_without_hex(gNetworkPlayers[i].name)
                if string.len(name) > 10 then
                    djui_hud_print_text(string.sub(name, 0, 10) .. "...", 4, j*30, 1)
                else
                    djui_hud_print_text(name, 4, j*30, 1)
                end
                djui_hud_set_color(255, 255, 64, 255)
                djui_hud_print_text(tostring(gPlayerSyncTable[i].seekerScore + gPlayerSyncTable[i].hiderScore), 180, j*30, 1)
                djui_hud_set_color(128, 128, 255, 255)
                djui_hud_print_text(tostring(gPlayerSyncTable[i].hiderScore), 270, j*30, 1)
                djui_hud_set_color(255, 128, 128, 255)
                djui_hud_print_text(tostring(gPlayerSyncTable[i].seekerScore), 360, j*30, 1)
            end
        end
    else
        
        local hidersRemaining = 0
        local textRow = 0

        for i = 0, MAX_PLAYERS - 1 do 
            if gNetworkPlayers[i].connected and not gPlayerSyncTable[i].seeking then
                hidersRemaining = hidersRemaining + 1
            end
        end
        djui_hud_render_rect(0, 0, 200, 30*(connCount+6))
        djui_hud_set_color(255, 255, 255, 255)
        djui_hud_set_color(255, 255, 255, 255)
        if gGlobalSyncTable.rankToggle then
            if gGlobalSyncTable.levelList == 1 then
                djui_hud_print_text("Round " .. gGlobalSyncTable.levelIndex .. "/" .. #(standardLevels), 4, 0, 1)
            else
                djui_hud_print_text("Round " .. gGlobalSyncTable.levelIndex .. "/" .. #(allLevels), 4, 0, 1)
            end
            textRow = textRow + 30
        end 
        
        
        djui_hud_print_text("Players: " .. connCount, 4, textRow, 1)
        textRow = textRow + 60

        djui_hud_set_color(192, 192, 255, 255)
        djui_hud_print_text("Hiders (" .. connCount - seekerCount .. ")", 4, textRow, 1)
        textRow = textRow + 30
        
        for i = 0, MAX_PLAYERS - 1 do 
            if gNetworkPlayers[i].connected and not gPlayerSyncTable[i].seeking then

                local r, g, b = hex_to_rgb(network_get_player_text_color_string(i))
                djui_hud_set_color(r, g, b, 255)

                local name = string_without_hex(gNetworkPlayers[i].name)
                if string.len(name) > 10 then
                    djui_hud_print_text(string.sub(name, 0, 10) .. "...", 4, textRow, 1)
                else
                    djui_hud_print_text(name, 4, textRow, 1)
                end

                textRow = textRow + 30

            end
        end
        textRow = textRow + 30
        djui_hud_set_color(255, 128, 128, 255)
        djui_hud_print_text("Seekers (" .. seekerCount .. ")", 4, textRow, 1)
        textRow = textRow + 30
        
        for i = 0, MAX_PLAYERS - 1 do 
            if gNetworkPlayers[i].connected and gPlayerSyncTable[i].seeking then
                
                local r, g, b = hex_to_rgb(network_get_player_text_color_string(i))
                djui_hud_set_color(r, g, b, 255)

                local name = string_without_hex(gNetworkPlayers[i].name)
                if string.len(name) > 10 then
                    djui_hud_print_text(string.sub(name, 0, 10) .. "...", 4, textRow, 1)
                else
                    djui_hud_print_text(name, 4, textRow, 1)
                end

                textRow = textRow + 30

            end
        end

    end

    sFlashingIndex = sFlashingIndex + 1
end

function string_without_hex(name)
    local s = ''
    local inSlash = false
    for i = 1, #name do
        local c = name:sub(i,i)
        if c == '\\' then
            inSlash = not inSlash
        elseif not inSlash then
            s = s .. c
        end
    end
    return s
end

function hex_to_rgb(hex)
	-- remove the # and the \\ from the hex so that we can convert it properly
	hex = hex:gsub('#','')
	hex = hex:gsub('\\','')

    -- sanity check
	if string.len(hex) == 6 then
		return tonumber('0x'..hex:sub(1,2)), tonumber('0x'..hex:sub(3,4)), tonumber('0x'..hex:sub(5,6))
	else
		return 0, 0, 0
	end
end

local function level_init()
    local s = gPlayerSyncTable[0]

    pauseExitTimer = 0
    canLeave = false

    if s.seeking then canLeave = true end
end

local function on_pause_exit()
    local s = gPlayerSyncTable[0]

    if not canLeave and not s.seeking then
        djui_popup_create(tostring(math_floor(30 - pauseExitTimer / 30)).." seconds until you can leave!", 2)
        return false
    end
end


-- REMOVE STAR SPAWN CUTSCENE

function remove_timestop()
    ---@type MarioState
    local m = gMarioStates[0]
    ---@type Camera
    local c = gMarioStates[0].area.camera

    if not m or not c then
        return
    end

    if ((c.cutscene == CUTSCENE_STAR_SPAWN) or (c.cutscene == CUTSCENE_RED_COIN_STAR_SPAWN) or (c.cutscene == CUTSCENE_ENTER_BOWSER_ARENA)) then
        print("disabled cutscene")
        disable_time_stop_including_mario()
        m.freeze = 0
        if c.cutscene == CUTSCENE_ENTER_BOWSER_ARENA then
          local bowser = obj_get_first_with_behavior_id(id_bhvBowser)
          if bowser and bowser.oAction == 5 then
            if bowser.oBehParams2ndByte == 0x01 then
              bowser.oAction = 13
            else
              bowser.oAction = 0
            end
            if m.action == ACT_READING_NPC_DIALOG then
              set_mario_action(m, ACT_IDLE, 0)
            end
          end
        elseif c.cutscene == CUTSCENE_STAR_SPAWN then -- done here because a lot of hacks hook to this object
          local grand = obj_get_first_with_behavior_id(id_bhvGrandStar)
          if grand then
            grand.oAction = 1
            m.invincTimer = 600 -- 20 seconds is long enough I think
            --obj_become_tangible(grand)
          end
        end
        c.cutscene = 0
        play_cutscene(c)
    elseif m.invincTimer < 30 and c.cutscene ~= 0 and gGlobalSyncTable.mhState == 2 then
        m.invincTimer = 30
    end
end


-- AUTOMATIC DOORS

function door_loop(o)
    o.collisionData = nil
    o.hitboxRadius = 0
    o.hitboxHeight = 0
    if o.oAction == 0 then
      -- if mario is close enough, set action to the custom open door action, 5
      if dist_between_objects(o, gMarioStates[0].marioObj) <= 400 then
          o.oAction = 5
      end
    end

    if o.oAction == 5 then
        if o.oTimer == 0 then
          -- when the object timer is 0 (when we first set the action to 5) play a sound and init the animation
          cur_obj_init_animation_with_sound(1)
    
          cur_obj_play_sound_2(SOUND_GENERAL_OPEN_WOOD_DOOR)
        end
    
        if o.header.gfx.animInfo.animFrame < 10 then
          o.header.gfx.animInfo.animFrame = 10 -- make door opening feel snappier
        end
    
        -- 40 is the anim frame where the door is fully opened
        if o.header.gfx.animInfo.animFrame >= 40 then
          o.header.gfx.animInfo.animFrame = 40
          o.header.gfx.animInfo.prevAnimFrame = 40
        end
    
        -- if we are far from the door, go to the custom close door action, 6
        if dist_between_objects(o, gMarioStates[0].marioObj) > 400 then
          o.oAction = 6
        end
      end
    
      if o.oAction == 6 then
        -- since the action is no longer 5, the animation continues, 78 is the end of the animation (take 2 frames)
        if o.header.gfx.animInfo.animFrame >= 78 then
          -- play object sound, and set action to 0
          cur_obj_play_sound_2(SOUND_GENERAL_CLOSE_WOOD_DOOR)
          o.oAction = 0
        end
      end
end

function star_door_loop(o)
    local m = gMarioStates[0]
    local starsNeeded = (o.oBehParams >> 24) or 0 -- this gets the star count
    if gGlobalSyncTable.freeRoam then
      starsNeeded = 0
    elseif gGlobalSyncTable.starRun and gGlobalSyncTable.starRun ~= -1 and gGlobalSyncTable.starRun <= starsNeeded then
      local np = gNetworkPlayers[0]
      starsNeeded = gGlobalSyncTable.starRun
      if (np.currAreaIndex ~= 2) and ROMHACK.ddd == true then
        starsNeeded = starsNeeded - 1
      end
    end
    if starsNeeded <= m.numStars and dist_between_objects(m.marioObj, o) <= 800 then
        o.oIntangibleTimer = -1
        if o.oAction == 0 then
          o.oAction = 1
          doorsClosing = false
        elseif o.oAction == 3 and not doorsClosing then
          o.oAction = 2
        end
        doorsCanClose = false
    elseif o.oAction == 3 then
        if doorsCanClose == false and not doorsClosing then
          o.oAction = 2
          doorsCanClose = true
        else
          doorsClosing = true
        end
    end
end

--DISABLE CANNONS

---@param o Object
local function cannon_lid_init(o)
    o.oFlags = OBJ_FLAG_PERSISTENT_RESPAWN | OBJ_FLAG_UPDATE_GFX_POS_AND_ANGLE
    o.collisionData = gGlobalObjectCollisionData.cannon_lid_seg8_collision_08004950
    cur_obj_set_home_once()
end

---@param o Object
local function cannon_lid_loop(o)
    if
        --WHITELIST LEVELS
        gGlobalSyncTable.forcedLevel == LEVEL_SSL or
        gGlobalSyncTable.forcedLevel == LEVEL_WDW or
        gGlobalSyncTable.forcedLevel == LEVEL_TTM or
        gGlobalSyncTable.forcedLevel == LEVEL_THI or
        gGlobalSyncTable.forcedLevel == LEVEL_RR or
        gGlobalSyncTable.forcedLevel == LEVEL_WMOTR
    then
        obj_set_model_extended(o, E_MODEL_NONE)
    else
        obj_set_model_extended(o, E_MODEL_DL_CANNON_LID)
        load_object_collision_model()
    end
end

id_bhvCannonLid = hook_behavior(nil, OBJ_LIST_SURFACE, false, cannon_lid_init, cannon_lid_loop, "cannonLid")
id_bhvCannonClosed = hook_behavior(id_bhvCannonClosed, OBJ_LIST_SURFACE, false, function (o)
    spawn_non_sync_object(id_bhvCannonLid, E_MODEL_DL_CANNON_LID, o.oPosX, o.oPosY - 5, o.oPosZ, function (obj)
        obj.oFaceAnglePitch = o.oFaceAnglePitch
        obj.oFaceAngleYaw = o.oFaceAngleYaw
        obj.oFaceAngleRoll = o.oFaceAngleRoll
    end)
    o.activeFlags = ACTIVE_FLAG_DEACTIVATED
end, nil, nil)

---@param o Object
local function hidden_120_init(o)
    o.oFlags = OBJ_FLAG_UPDATE_GFX_POS_AND_ANGLE
    o.collisionData = gGlobalObjectCollisionData.castle_grounds_seg7_collision_cannon_grill
    o.oCollisionDistance = 4000
end

---@param o Object
local function hidden_120_loop(o)
    obj_set_model_extended(o, E_MODEL_NONE)
end

hook_behavior(id_bhvHiddenAt120Stars, OBJ_LIST_SURFACE, true, hidden_120_init, hidden_120_loop)

-----------------------
-- network callbacks --
-----------------------

local function on_round_state_changed()
    local rs = gGlobalSyncTable.roundState

    if rs == ROUND_STATE_SEEKERS_WIN then
        play_sound(SOUND_MENU_CLICK_CHANGE_VIEW, gMarioStates[0].marioObj.header.gfx.cameraToObject)
    elseif rs == ROUND_STATE_HIDERS_WIN then
        play_sound(SOUND_MENU_CLICK_CHANGE_VIEW, gMarioStates[0].marioObj.header.gfx.cameraToObject)
    end
end

local function on_seeking_changed(tag, oldVal, newVal)
    local m = gMarioStates[tag]
    local npT = gNetworkPlayers[tag]

    -- play sound and create popup if became a seeker
    if newVal and not oldVal then
        playerColor = network_get_player_text_color_string(m.playerIndex)
        djui_popup_create(playerColor .. npT.name .. "\\#ffa0a0\\ is now a seeker!", 2)
        if gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
            if sRoundTimer - sRoundAddTime < 35 then
                sRoundTimer = 35
            else
                sRoundTimer = sRoundTimer - sRoundAddTime
            end
        end
    end

    if newVal then
        network_player_set_description(npT, "Seeker", 255, 128, 128, 255)
    else
        network_player_set_description(npT, "Hider", 128, 128, 255, 255)
    end
end

local function check_touch_tag_allowed(i)
    if gMarioStates[i].action ~= ACT_TELEPORT_FADE_IN and gMarioStates[i].action ~= ACT_TELEPORT_FADE_OUT and gMarioStates[i].action ~= ACT_PULLING_DOOR and gMarioStates[i].action ~= ACT_PUSHING_DOOR and gMarioStates[i].action ~= ACT_WARP_DOOR_SPAWN and gMarioStates[i].action ~= ACT_ENTERING_STAR_DOOR and gMarioStates[i].action ~= ACT_STAR_DANCE_EXIT and gMarioStates[i].action ~= ACT_STAR_DANCE_NO_EXIT and gMarioStates[i].action ~= ACT_STAR_DANCE_WATER and gMarioStates[i].action ~= ACT_PANTING and gMarioStates[i].action ~= ACT_UNINITIALIZED and gMarioStates[i].action ~= ACT_WARP_DOOR_SPAWN then
        return true
    end

    return false
end

local function on_interact(m, obj, intee)
    if intee == INTERACT_PLAYER then
        if m ~= gMarioStates[0] then
            for i=0,(MAX_PLAYERS-1) do
                if gNetworkPlayers[i].connected and gNetworkPlayers[i].currAreaSyncValid then
                    if gPlayerSyncTable[m.playerIndex].seeking and not gPlayerSyncTable[i].seeking and obj == gMarioStates[i].marioObj and check_touch_tag_allowed(i) then
                        if (gMarioStates[i].action == ACT_VERTICAL_WIND or gGlobalSyncTable.touchTag) and gGlobalSyncTable.roundState == ROUND_STATE_ACTIVE then
                            local seekerCount = 0
                            for i = 0, (MAX_PLAYERS-1) do
                                if gNetworkPlayers[i].connected then
                                    if gPlayerSyncTable[i].seeking then
                                        seekerCount = seekerCount + 1
                                    end
                                end
                            end
                            gPlayerSyncTable[m.playerIndex].seekerScore = gPlayerSyncTable[m.playerIndex].seekerScore + seekerCount
                            gPlayerSyncTable[i].hiderScore = gPlayerSyncTable[i].hiderScore + seekerCount
                            gPlayerSyncTable[i].seeking = true
                            network_player_set_description(gNetworkPlayers[i], "Seeker", 255, 128, 128, 255)
                        end
                    end
                end
            end
        end
    end
end

local function allow_interact(_, _, intee)
    if intee == INTERACT_KOOPA_SHELL and gGlobalSyncTable.banKoopaShell then
        return false
    end
end

function allow_pvp_attack(m1, m2)
    local s1 = gPlayerSyncTable[m1.playerIndex]
    local s2 = gPlayerSyncTable[m2.playerIndex]

    if(
        --Disables Team Attack
        s1.seeking == s2.seeking or 
        --Hiders Cannot Attack
        not s1.seeking or 
        --Cannot Attack During Intermission
        gGlobalSyncTable.roundState ~= ROUND_STATE_ACTIVE 
    )
    then
        return false
    end

    return true

end

-----------
-- hooks --
-----------

hook_event(HOOK_UPDATE, update)
hook_event(HOOK_ON_SCREEN_TRANSITION, screen_transition)
hook_event(HOOK_BEFORE_SET_MARIO_ACTION, before_set_mario_action)
hook_event(HOOK_BEFORE_MARIO_UPDATE, before_mario_update)
hook_event(HOOK_MARIO_UPDATE, mario_update)
hook_event(HOOK_ALLOW_PVP_ATTACK, allow_pvp_attack)
hook_event(HOOK_ON_PVP_ATTACK, on_pvp_attack)
hook_event(HOOK_ON_PLAYER_CONNECTED, on_player_connected)
hook_event(HOOK_ON_HUD_RENDER, on_hud_render)
hook_event(HOOK_ON_LEVEL_INIT, level_init)
hook_event(HOOK_ON_PAUSE_EXIT, on_pause_exit) -- timer
hook_event(HOOK_ON_INTERACT, on_interact)
hook_event(HOOK_ALLOW_INTERACT, allow_interact)
hook_event(HOOK_USE_ACT_SELECT, function () return false end)
hook_event(HOOK_UPDATE, remove_timestop)
hook_behavior(id_bhvDoor, OBJ_LIST_SURFACE, false, door_init, door_loop)
hook_behavior(id_bhvStarDoor, OBJ_LIST_SURFACE, false, nil, star_door_loop)

-- HOST COMMANDS

local function command_on_touch_tag()
    gGlobalSyncTable.touchTag = not gGlobalSyncTable.touchTag
    djui_chat_message_create("Touch to Tag: " .. on_or_off(gGlobalSyncTable.touchTag))
    return true
end

local function command_all_levels_toggle()
    gGlobalSyncTable.levelIndex = 0
    if gGlobalSyncTable.levelList == 1 then
        gGlobalSyncTable.levelList = 2
        djui_chat_message_create("Switched to All Levels Mode. Next round will start in the Castle Courtyard.")
    else gGlobalSyncTable.levelList = 1
        djui_chat_message_create("Switched to Standard Levels Mode. Next round will start inside the Castle.")
    end
    return true
end

local function command_make_all_seekers()
    for i=0,(MAX_PLAYERS-1) do
        gPlayerSyncTable[i].seeking = true
    end
    return true
end

local function command_force_finish()
    sRoundTimer = sRoundEndTimeout
    return true
end

local function command_toggle_scoreboard()
    gGlobalSyncTable.rankToggle = not gGlobalSyncTable.rankToggle
    return true
end

local function command_reset_scores()
    for i = 0, MAX_PLAYERS - 1 do
        gPlayerSyncTable[i].hiderScore = 0
    end
    for i = 0, MAX_PLAYERS - 1 do
        gPlayerSyncTable[i].seekerScore = 0
    end
    djui_chat_message_create("All scores have been reset.")
    return true
end


if network_is_server() then
   hook_chat_command("touch-to-tag", "- Turn touch tag on or off.", command_on_touch_tag)
   hook_chat_command("all-levels", "- Toggle all levels to be playable.", command_all_levels_toggle)
   hook_chat_command("all-seekers", "- Make everyone a seeker (use responsibly.)", command_make_all_seekers)
   hook_chat_command("f", "- Forcibly end a round.", command_force_finish)
   hook_chat_command("ranked", "- Toggle the scoreboard's visibility.", command_toggle_scoreboard)
   hook_chat_command("reset-scores", "- Reset all scores on the scoreboard.", command_reset_scores)
end

-- call functions when certain sync table values change
hook_on_sync_table_change(gGlobalSyncTable, "roundState", 0, on_round_state_changed)

for i = 0, (MAX_PLAYERS - 1) do
    gPlayerSyncTable[i].seeking = true
    hook_on_sync_table_change(gPlayerSyncTable[i], "seeking", i, on_seeking_changed)
    network_player_set_description(gNetworkPlayers[i], "Seeker", 255, 128, 128, 255)
end

_G.HideAndSeek = {
    is_player_seeker = function (playerIndex)
        return gPlayerSyncTable[playerIndex].seeking
    end,

    set_player_seeker = function (playerIndex, seeking)
        gPlayerSyncTable[playerIndex].seeking = seeking
    end,
}
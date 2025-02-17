require("hs.ipc")

-- Smooth scrolling variables
local scrollSpeed = 5
local scrollTimer = nil
local scrollAmount = 0

-- Function to smoothly scroll
function smoothScroll(amount)
    if scrollTimer then scrollTimer:stop() end
    scrollAmount = amount
    scrollTimer = hs.timer.new(0.01, function()
        hs.eventtap.scrollWheel({0, scrollAmount}, {}, "pixel")
    end)
    scrollTimer:start()
end

-- Function to stop scrolling
function stopScroll()
    if scrollTimer then
        scrollTimer:stop()
    end
    scrollAmount = 0
end

-- Expose global functions for Python to call
function startScrollDown()
    smoothScroll(-scrollSpeed)
end

function startScrollUp()
    smoothScroll(scrollSpeed)
end

-- Keep your original hotkeys if you want
hs.hotkey.bind({}, "F6", function() smoothScroll(-scrollSpeed) end, stopScroll)
hs.hotkey.bind({}, "F8", function() smoothScroll(scrollSpeed) end, stopScroll)

hs.alert.show("Hammerspoon Smooth Scroll Loaded!")
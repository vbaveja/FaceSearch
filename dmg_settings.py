import os

# Use the current working directory as the project root
app_root = os.getcwd()

# Name of the volume when mounted
volume_name = "FaceSearch"

# What to include inside the DMG
files = [
    os.path.join(app_root, "dist", "FaceSearch.app"),
]

# Symlinks (shortcuts) inside the DMG
symlinks = {
    "Applications": "/Applications",
}

# DMG format: compressed, read-only
format = "UDZO"

# Optional layout settings
background = None  # you can set a PNG later if you want a background
window_rect = ((200, 200), (600, 400))
icon_size = 96

icon_locations = {
    "FaceSearch.app": (150, 200),
    "Applications": (450, 200),
}

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SubwayBuilder Patcher is a modular game modification tool for the Subway Builder Electron game. It extracts the game's asar archive, applies patches via a plugin system ("packages"), and repackages the game. Cross-platform support for Windows, Linux (AppImage), and macOS.

## Commands

### Run the Patcher

**GUI Mode (Recommended):**
```bash
# Windows
.\Start_GUI.bat

# Linux/macOS
./Start_GUI.sh
```
Opens http://localhost:3000 with a 4-step wizard.

**CLI Mode:**
```bash
node ./patcher/patch_game.js
```
Requires `config.js` to be configured first.

### Install Dependencies
```bash
# Windows
.\Install\ dependencies.bat

# Linux/macOS
./Install\ dependencies.sh
```

### Map Patcher Utilities
```bash
cd patcher/packages/mapPatcher

# Download OSM data for configured cities
node download_data.js

# Process raw data into game format
node process_data.js

# Download map tiles
node download_tiles.js
```

## Architecture

### Patching Flow

1. Copy game to `patching_working_directory/`
2. Extract `app.asar` using @electron/asar
3. Load file contents: `INDEX`, `GAMEMAIN`, `INTERLINEDROUTES`, `POPCOMMUTEWORKER` from `dist/renderer/public/`
4. Run packages **sequentially** (order matters) - each receives modified content from previous
5. Write modified files back
6. Repack asar and create output (`SubwayBuilderPatched/` on Windows)

### Package System

Packages live in `patcher/packages/[name]/` and must export a `patcherExec` function:

```javascript
// patcher/packages/[name]/patcherExec.js
export function patcherExec(fileContents) {
  // fileContents has: INDEX, GAMEMAIN, INTERLINEDROUTES, POPCOMMUTEWORKER, PATHS
  // Modify via string/regex replacement
  // Return modified fileContents (or Promise for async)
  return fileContents;
}
```

### Built-in Packages

- **mapPatcher** - Adds custom cities from OpenStreetMap data. Most complex - downloads data, processes tiles, generates thumbnails. Async/Promise-based.
- **addTrains** - Adds custom train types. Config-driven with validation. Patches TRAIN_TYPES object and cost calculations.
- **devTools** - Injects DevTools opener into main process. Hooks `browser-window-created` event.

### Key Files

- `patcher/patch_game.js` - Main orchestrator
- `gui_server.js` - Express server for web wizard
- `config.js` - User's patcher config (gitignored, copy from `config_[platform].js`)
- `public/client.js` - GUI wizard client logic

## Configuration

Root config (`config.js`):
```javascript
const config = {
  "subwaybuilderLocation": "C:\\...\\Subway Builder\\",
  "platform": "windows", // or "linux", "macos"
  "packagesToRun": ["addTrains", "devTools", "mapPatcher"]
};
export default config;
```

Package configs are in their respective directories (e.g., `patcher/packages/mapPatcher/config.js`).

## Important Notes

- Game files are dynamically hashed (`index-[hash].js`). The patcher finds them by prefix.
- Packages modify minified JavaScript via regex - fragile if game updates obfuscation.
- Main process code (`dist/main/main.jsc`) is V8 bytecode. To inject code, modify the loader (`dist/main/main.js`) before it loads the bytecode.
- Linux requires `appimagetool` installed separately.
- macOS output is signed with ad-hoc certificate (`codesign -s -`).

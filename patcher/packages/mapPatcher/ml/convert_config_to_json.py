"""
Convert config.js to config.json

Utility script to convert the JavaScript config file to JSON format
that can be used by both Python and Node.js.

Usage: python convert_config_to_json.py
"""

import json
import re
from pathlib import Path


def parse_js_config(js_content: str) -> dict:
    """Parse JavaScript config file to extract configuration"""
    
    # Extract the config object
    config_match = re.search(r'const config = ({.*?});', js_content, re.DOTALL)
    if not config_match:
        raise ValueError("Could not find config object in JS file")
    
    config_str = config_match.group(1)
    
    # Replace JavaScript syntax with JSON-compatible syntax
    # Handle comments
    config_str = re.sub(r'//.*?\n', '\n', config_str)
    config_str = re.sub(r'/\*.*?\*/', '', config_str, flags=re.DOTALL)
    
    # Replace single quotes with double quotes (simple approach)
    config_str = config_str.replace("'", '"')
    
    # Remove trailing commas
    config_str = re.sub(r',(\s*[}\]])', r'\1', config_str)
    
    # Try to parse as JSON
    try:
        config = json.loads(config_str)
        return config
    except json.JSONDecodeError as e:
        print(f"Failed to parse as JSON: {e}")
        print("Attempting manual extraction...")
        
        # Fallback: manual extraction
        config = {}
        
        # Extract simple key-value pairs
        for match in re.finditer(r'"([^"]+)":\s*([^,\n}]+)', config_str):
            key = match.group(1)
            value_str = match.group(2).strip()
            
            # Try to convert value
            try:
                if value_str == 'true':
                    value = True
                elif value_str == 'false':
                    value = False
                elif value_str == 'null':
                    value = None
                elif value_str.replace('.', '').replace('-', '').isdigit():
                    value = float(value_str) if '.' in value_str else int(value_str)
                else:
                    value = value_str.strip('"')
                
                config[key] = value
            except:
                pass
        
        # Extract places array separately
        places_match = re.search(r'"places":\s*\[(.*?)\]', config_str, re.DOTALL)
        if places_match:
            places_str = '[' + places_match.group(1) + ']'
            try:
                config['places'] = json.loads(places_str)
            except:
                print("Warning: Could not parse places array")
        
        return config


def main():
    """Convert config.js to config.json"""
    script_dir = Path(__file__).parent
    map_patcher_dir = script_dir.parent
    
    config_js_path = map_patcher_dir / 'config.js'
    config_json_path = map_patcher_dir / 'config.json'
    
    if not config_js_path.exists():
        print(f"Error: {config_js_path} not found")
        return
    
    print(f"Reading {config_js_path}...")
    with open(config_js_path, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    print("Parsing JavaScript config...")
    try:
        config = parse_js_config(js_content)
    except Exception as e:
        print(f"Error parsing config: {e}")
        return
    
    print(f"Writing to {config_json_path}...")
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Successfully converted config.js to config.json")
    print(f"  Parameters: {len([k for k in config.keys() if k != 'places'])}")
    print(f"  Places: {len(config.get('places', []))}")
    
    if config.get('places'):
        print(f"  Cities: {', '.join(p['code'] for p in config['places'])}")


if __name__ == '__main__':
    main()

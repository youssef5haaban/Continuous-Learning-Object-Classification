import json

category_mapping = {
    0: 'plug_adapter', 1: 'mobile_phone', 2: 'scissor', 3: 'light_bulb', 4: 'can', 5: 'glass',
    6: 'ball', 7: 'marker', 8: 'cup', 9: 'remote_control', 10: 'glass', 11: 'ball', 12: 'marker',
    13: 'cup', 14: 'remote_control', 15: 'plug_adapter', 16: 'mobile_phone', 17: 'scissor',
    18: 'light_bulb', 19: 'can', 20: 'glass', 21: 'ball', 22: 'marker', 23: 'cup',
    24: 'remote_control', 25: 'plug_adapter', 26: 'mobile_phone', 27: 'scissor', 28: 'light_bulb',
    29: 'can', 30: 'plug_adapter', 31: 'mobile_phone', 32: 'scissor', 33: 'light_bulb',
    34: 'can', 35: 'plug_adapter', 36: 'mobile_phone', 37: 'scissor', 38: 'light_bulb',
    39: 'can', 40: 'glass', 41: 'ball', 42: 'marker', 43: 'cup', 44: 'remote_control',
    45: 'glass', 46: 'ball', 47: 'marker', 48: 'cup', 49: 'remote_control'
}

with open('category_mapping.txt', 'w') as file:
    json.dump(category_mapping, file)
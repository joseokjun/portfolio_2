#-*-coidng:utf-8-*-
import os

class Config:
    APP_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.dirname(APP_DIR)

    # database info
    DB_HOST = "59.25.177.42"
    DB_USER = "smartfarm"
    DB_PW = 'smartfarm1!'
    DATABASE = "smartfarm"
    DB_PORT = 3355

    # 변수 선택
    custom_x_strawberry = {'sky_window':
['tp_y', 'sr', 'hd_x', 'pc', 'tp_x', 'wd', 'co2', 'ws'],
'cg_curtain':
['tp_y', 'sr', 'hd_x', 'pc', 'co2', 'tp_x', 'ws', 'wd'],
'flow_fan':
['tp_y', 'tp_x', 'sr', 'co2', 'pc', 'hd_x', 'ws', 'wd'],
'hmdfc':
['tp_y', 'hd_x', 'sr', 'co2', 'tp_x', 'ws', 'wd'],
'crc_pump':
['tp_y', 'sr', 'hd_x', 'tp_x', 'co2', 'pc'],
'3way_valve':
['tp_y', 'sr', 'hd_x', 'tp_x', 'co2', 'pc'],
'heat_cooler':
['hd_x', 'co2', 'sr', 'tp_x', 'tp_y', 'pc']}

    custom_x_tomato = {'sky_window':
['tp_y', 'sr', 'hd_x', 'ws', 'tp_x', 'co2'],
'cg_curtain':
['tp_y', 'sr', 'tp_x', 'hd_x', 'pc', 'ws', 'co2'],
'flow_fan':
['sr', 'tp_x', 'ws', 'hd_x', 'pc', 'co2', 'tp_y'],
'hmdfc':
['tp_x', 'tp_y', 'hd_x', 'sr', 'ws'],
'crc_pump':
['tp_y', 'tp_x', 'hd_x', 'pc', 'co2', 'ws', 'wd'],
'3way_valve':
['tp_y', 'tp_x', 'hd_x', 'pc', 'co2', 'ws', 'wd'],
'heat_cooler':
['tp_y', 'tp_x', 'hd_x', 'co2', 'sr', 'pc', 'wd']}

    custom_x_melon = {'sky_window':
['tp_y', 'tp_x', 'sr', 'hd_x', 'ws', 'wd'],
'cg_curtain':
['tp_x', 'tp_y', 'hd_x', 'ws', 'co2', 'pc', 'sr'],
'flow_fan':
['tp_y', 'tp_x', 'ws', 'sr', 'co2', 'pc', 'hd_x'],
'hmdfc':
['tp_y', 'sr', 'tp_x', 'hd_x', 'co2', 'ws'],
'crc_pump':
['tp_y', 'tp_x', 'hd_x', 'ws', 'sr'],
'3way_valve':
['tp_y', 'hd_x', 'sr', 'ws', 'co2', 'wd', 'tp_x'],
'heat_cooler':
['tp_y', 'hd_x', 'sr', 'co2']}

# 실제 만다린이 아니고 멜론임!
    custom_x_mandarin = {'sky_window':
['tp_y', 'tp_x', 'sr', 'hd_x', 'ws', 'wd'],
'cg_curtain':
['tp_x', 'tp_y', 'hd_x', 'ws', 'co2', 'pc', 'sr'],
'flow_fan':
['tp_y', 'tp_x', 'ws', 'sr', 'co2', 'pc', 'hd_x'],
'hmdfc':
['tp_y', 'sr', 'tp_x', 'hd_x', 'co2', 'ws'],
'crc_pump':
['tp_y', 'tp_x', 'hd_x', 'ws', 'sr'],
'3way_valve':
['tp_y', 'hd_x', 'sr', 'ws', 'co2', 'wd', 'tp_x'],
'heat_cooler':
['tp_y', 'hd_x', 'sr', 'co2']}


    custom_x = {'strawberry':custom_x_strawberry,'tomato':custom_x_tomato,'melon':custom_x_melon,'mandarin':custom_x_mandarin}
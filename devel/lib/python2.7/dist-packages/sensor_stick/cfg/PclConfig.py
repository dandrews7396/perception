## *********************************************************
## 
## File autogenerated for the sensor_stick package 
## by the dynamic_reconfigure package.
## Please do not edit.
## 
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'upper': 'DEFAULT', 'lower': 'groups', 'srcline': 246, 'name': 'Default', 'parent': 0, 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'cstate': 'true', 'parentname': 'Default', 'class': 'DEFAULT', 'field': 'default', 'state': True, 'parentclass': '', 'groups': [], 'parameters': [{'srcline': 293, 'description': 'Select Axis', 'max': 2, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'axis', 'edit_method': "{'enum_description': 'Select axis', 'enum': [{'srcline': 8, 'description': 'X-axis', 'srcfile': '/home/robond/perception/src/sensor_stick/config/Pcl.cfg', 'cconsttype': 'const int', 'value': 0, 'ctype': 'int', 'type': 'int', 'name': 'X'}, {'srcline': 9, 'description': 'Y-axis', 'srcfile': '/home/robond/perception/src/sensor_stick/config/Pcl.cfg', 'cconsttype': 'const int', 'value': 1, 'ctype': 'int', 'type': 'int', 'name': 'Y'}, {'srcline': 10, 'description': 'Z-axis', 'srcfile': '/home/robond/perception/src/sensor_stick/config/Pcl.cfg', 'cconsttype': 'const int', 'value': 2, 'ctype': 'int', 'type': 'int', 'name': 'Z'}]}", 'default': 2, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 293, 'description': 'min', 'max': 3.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'min', 'edit_method': '', 'default': 0.77, 'level': 0, 'min': -0.5, 'type': 'double'}, {'srcline': 293, 'description': 'max', 'max': 3.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'max', 'edit_method': '', 'default': 2.0, 'level': 0, 'min': -0.5, 'type': 'double'}], 'type': '', 'id': 0}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])    
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

Pcl_X = 0
Pcl_Y = 1
Pcl_Z = 2

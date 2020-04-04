'''Validation related functions.'''

import pkg_resources
import json
import jsonref
import jsonschema

#########################################
def validate_json_with_schema_file(loaded_json, schema_name):
    '''
    Validate a loaded JSON object using a schema file in the json_schema resource directory.
    
    :param str loaded_json: The JSON object.
    :param str schema_name: The file name of the json_schema resource.
    :raise jsonschema.exceptions.ValidationError: Invalid JSON.
    '''
    schema = jsonref.loads(
        pkg_resources.resource_string('asemi_segmenter.resources.json_schema', schema_name).decode(),
        base_uri='file:///{}/'.format(
            pkg_resources.resource_filename('asemi_segmenter.resources.json_schema', '')
            ),
        jsonschema=True
        )
    jsonschema.Draft7Validator.check_schema(schema)
    validator = jsonschema.Draft7Validator(schema)
    validator.validate(loaded_json)
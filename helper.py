import os
import json
import gzip
DEFAULT_SEED = 9999

def save_json_file(json_obj, path):
    create_folder_for_file(path)
    f = open(path, 'w')
    json.dump(json_obj, f, indent=4)
    return

def open_file(fn):
	if not os.path.isfile(fn):
		print ( "File: " + fn + " DOES NOT EXISTS")
		exit(1)
	if fn.endswith('.gz'):
		return gzip.open(fn, 'rt')
	else:
		return open(fn, 'r')

def check_colnames_in_df(columns_to_check, df):
	"""
	:param columns_to_check: list of columns that we want to check to be present in a dataframe df
	:param df: the dataframe
	:return: if there are missing columns in columns to check, we will need to terminate the program
	"""
	assert set(columns_to_check).issubset(df.columns), ('Columns specified by users are NOT ALL present '
														'in the input dataframes: {}').format(columns_to_check)

def make_dir(directory):
	try:
		os.makedirs(directory)
	except:
		pass



def check_file_exist(fn):
	if not os.path.isfile(fn):
		print ( "File: " + fn + " DOES NOT EXISTS")
		exit(1)
	return 

def check_dir_exist(fn):
	if not os.path.isdir(fn):
		print ( "Directory: " + fn + " DOES NOT EXISTS")
		exit(1)
	return 
	
def create_folder_for_file(fn):
	last_slash_index = fn.rfind('/')
	if last_slash_index != -1: # path contains folder
		make_dir(fn[:last_slash_index])
	return 

def get_command_line_float(arg):
	try: 
		arg = float(arg)
		return arg
	except:
		print ( "Integer: " + str(arg) + " IS NOT VALID")
		exit(1)


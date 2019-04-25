from util.CustomAgent import CustomAgent
from util.CustomLogger import CustomLogger
from netsapi.challenge import *
import sys, os, numpy as np, pandas as pd, logging, shutil
from distutils.dir_util import copy_tree

_submission_dir = 'submission'
_util_dir = 'util'
_output_csv = 'test.csv'

logger = CustomLogger(__name__)

#
# Main
#

logger.info('start running')
env = ChallengeEnvironment(experimentCount = 1000)
a = CustomAgent(env, 2)
a.scoringFunction()

logger.info('create submissions file {}'.format(_output_csv))
a.create_submissions(_output_csv)

#
# Move & Copy File
#

# create new submission folder
_submission_index = os.listdir('submission')
_submission_index.sort()
_submission_index = '1' if len(_submission_index) == 0 else str(int(_submission_index[-1]) + 1)
_new_submission_dir = os.path.join(_submission_dir, _submission_index)
os.makedirs(_new_submission_dir)
os.makedirs(os.path.join(_submission_dir, _submission_index,_util_dir))

# move scripts and files into submission folder
shutil.move(_output_csv, _new_submission_dir)
shutil.move(logger._log_file, _new_submission_dir)
copy_tree(_util_dir, os.path.join(_new_submission_dir,_util_dir))
shutil.copy(os.path.basename(__file__), _new_submission_dir)

logger.info('end running')
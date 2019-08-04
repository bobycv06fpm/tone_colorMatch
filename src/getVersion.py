import subprocess
import os

#NOTE this is only going to work when the process is run directly from the git repo
def getVersion(path):
     #git --git-dir=/home/dmacewen/Projects/tone/tone_colorMatch/.git/ --work-tree=/home/dmacewen/Projects/tone/tone_colorMatch tag
    gitDirPath = os.path.join(path, '.git')
    print('Looking for version info in :: {}'.format(gitDirPath))
    tagCommit = subprocess.run(["git", "--git-dir={}".format(gitDirPath), "--work-tree={}".format(path), "rev-list", "--tags", "--max-count=1"], stdout=subprocess.PIPE).stdout.decode().split('\n', 1)[0]                                               
    return subprocess.run(["git", "--git-dir={}".format(gitDirPath), "--work-tree={}".format(path), "describe", "--tags", tagCommit], stdout=subprocess.PIPE).stdout.decode().split('\n', 1)[0]                                                                 


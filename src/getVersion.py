import subprocess

#NOTE this is only going to work when the process is run directly from the git repo
def getVersion():
    tagCommit = subprocess.run(["git", "rev-list", "--tags", "--max-count=1"], stdout=subprocess.PIPE).stdout.decode().split('\n', 1)[0]                                               
    return subprocess.run(["git", "describe", "--tags", tagCommit], stdout=subprocess.PIPE).stdout.decode().split('\n', 1)[0]                                                                 


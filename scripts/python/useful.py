

class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2



def tail(file, window=10):
    #Returns the last `window` lines of file `f` as a list.

    if window == 0:
        return []
    BUFSIZ = 1024
    f=open(file,'r')
    f.seek(0, 2)
    bytes = f.tell()
    size = window + 1
    block = -1
    data = []
    while size > 0 and bytes > 0:
        if bytes - BUFSIZ > 0:
            # Seek back one whole BUFSIZ
            f.seek(block * BUFSIZ, 2)
            # read BUFFER
            data.insert(0, f.read(BUFSIZ))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            data.insert(0, f.read(bytes))
        linesFound = data[0].count('\n')
        size -= linesFound
        bytes -= BUFSIZ
        block -= 1
    return ''.join(data).splitlines()[-window:]

def head(file, window=10):
    # Returns the first `window` lines of `f` as a list.
    
    data = []
    if window == 0:
        return []
    else:
        f=open(file,'r')
        tempCounter = 0
        for line in f:
            data.append(line[:-1])
            tempCounter +=1
            if tempCounter > window:
                break
        return data



import os, sys

def print_and_run(command, submit=False):
    print command
    if submit:
        os.system(command)

from glob import glob
def globs(string):
    if '}' not in string:
        return sorted(glob(string))
    else:
        start = string.split('{')[0]

        #blah{thing,that}lol{me,you}woo
        
        opens = string.split('{')[1:]
        contents = [o.split('}')[0].split(',') for o in opens]
        closes = string.split('}')[1:]
        inters = [c.split('{')[0] for c in closes]

        strings = []
        stringtemp = start
        for i in range(len(contents)):
            stringtemp += 'CONT'+str(i)+inters[i]

        stringtemps = []
        for i in range(len(contents)):
            stringtemps.append()
        print stringtemp
            

        print start, contents, inters
        print len(contents), len(inters)
        #opens = [blah]

def printdict(dict, isInt=False, printCounter=True):
    counter = 0
    if isInt:
        keys = dict.keys()
        keys = [int(key) for key in keys]
        keys.sort()
        keys = [str(key) for key in keys]
        
        for d in keys:
            counter +=1
            if printCounter: print counter, 
            print d, dict[d]

    else:
        for d in sorted(dict):
            counter +=1
            print counter, d, dict[d]

def printdictdict(dict, isInt=False, printCounter=True):
    for d in sorted(dict):
        print ''
        print d
        printdict(dict[d], isInt, printCounter)


def float_kMG(number, dp=0):
    decimalstring = "%."+str(dp)+"f"
    if number < 1000: return decimalstring % number + ' '
    elif number < 1000000: return (decimalstring %(number / 1000.0))+' k'
    elif number < 1000000000: return (decimalstring % (number / 1000000.0))+' M'
    elif number < 1000000000000: return (decimalstring % (number / 1000000000.0))+' G'
    else: return (decimalstring % (number / 1000000000000.0))+' T'

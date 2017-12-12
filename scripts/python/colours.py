



def print_format_table():
    """
    prints table of formatted text format options
    """
    for style in xrange(9):
        for fg in range(30,39)+range(90,99):
            s1 = ''
            for bg in xrange(40,48):
                format = ';'.join([str(style), str(fg), str(bg)])
                s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
            print s1
        print '\n'


def printcol(text, fgcol='white', style='normal', bgcol='none'):
    fgcols = {
        'dgrey':   2,
        'ddgrey':  8,

        'black':   30,
        'dred':    31,
        'dgreen':  32,
        'dyellow': 33,
        'dblue':   34,
        'dpink' :  35,
        'dcyan':   36,

        'pgrey':   37,
        'white':   38,

        'grey':    90,
        'red':     91,
        'green':   92,
        'yellow':  93,
        'blue':    94,
        'pink' :   95,
        'cyan':    96,
        }


    bgcols = {
        'none':    40,
        'red':     41,
        'green':   42,
        'yellow':  43,
        'blue':    44,
        'pink' :   45,
        'cyan':    46,
        'grey':    47,
        }


    styles = {
        'normal': 0,
        'bold': 1,
        'faded': 2,
        'underlined': 4,
        'flashing': 5,
        'fgbgrev': 7,
        'invisible': 8,
        }

    st = styles[style]
    fg = fgcols[fgcol]
    bg = bgcols[bgcol]

    format = ';'.join([str(st), str(fg), str(bg)])
    colstring = '\x1b[%sm%s\x1b[0m' % (format, text) 
    return colstring


#print_format_table()
#printcol("hi", "red", "bold", "blue")
#printcol("Wassup blud?", "blue", "flashing")

